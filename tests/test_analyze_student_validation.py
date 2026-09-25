import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fingertip_depth.student_training import (
    TRAINING_RUN_FORMAT,
    TRAINING_RUN_FORMAT_VERSION,
    regression_metrics,
)
from fingertip_depth.video_cache import sha256_file

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "analyze_student_validation.py"
_SPEC = importlib.util.spec_from_file_location("analyze_student_validation_for_test", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load {_SCRIPT_PATH}")
analyze_student_validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_student_validation)

_ROWS = (
    ("a:000000:hand0:aug=identity", "a", 0, 0.10, 0.11),
    ("a:000001:hand0:aug=identity", "a", 1, 0.20, 0.18),
    ("a:000003:hand0:aug=identity", "a", 3, 0.40, 0.39),
    ("b:000004:hand0:aug=identity", "b", 4, 0.25, 0.26),
    ("b:000005:hand0:aug=identity", "b", 5, 0.35, 0.37),
)


def _write_fixture(tmp_path: Path, *, corrupt_error: bool = False) -> Path:
    predictions_path = tmp_path / "validation_predictions_filtered.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "sample_id",
                "source_sequence_id",
                "frame_index",
                "augmentation_variant",
                "target_m",
                "prediction_m",
                "error_m",
                "absolute_error_m",
            ),
        )
        writer.writeheader()
        for index, (sample_id, sequence_id, frame_index, target_m, prediction_m) in enumerate(
            _ROWS
        ):
            error_m = prediction_m - target_m
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "source_sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "augmentation_variant": "identity",
                    "target_m": target_m,
                    "prediction_m": prediction_m,
                    "error_m": error_m + (0.01 if corrupt_error and index == 0 else 0.0),
                    "absolute_error_m": abs(error_m),
                }
            )

    targets = np.asarray([row[3] for row in _ROWS], dtype=np.float64)
    predictions = np.asarray([row[4] for row in _ROWS], dtype=np.float64)
    metrics = regression_metrics(predictions, targets)
    delta_mae_m = float(
        np.mean(
            [
                abs((0.18 - 0.11) - (0.20 - 0.10)),
                abs((0.37 - 0.26) - (0.35 - 0.25)),
            ]
        )
    )
    metrics["consecutive_frame_delta"] = {
        "consecutive_pair_count": 2,
        "delta_mae_m": delta_mae_m,
        "delta_mae_cm": 100.0 * delta_mae_m,
    }
    included_validation_path = tmp_path / "included_validation.txt"
    included_validation_path.write_text("".join(f"{row[0]}\n" for row in _ROWS), encoding="utf-8")
    included_validation_artifact = {
        "relative_path": included_validation_path.name,
        "sha256": sha256_file(included_validation_path),
    }
    artifact = {
        "relative_path": predictions_path.name,
        "sha256": sha256_file(predictions_path),
    }
    manifest = {
        "format": TRAINING_RUN_FORMAT,
        "format_version": TRAINING_RUN_FORMAT_VERSION,
        "task": {"target": "Depth Pro pseudo-label z_teacher_m", "ground_truth": False},
        "dataset": {
            "raw_validation_samples": len(_ROWS),
            "selected_validation_samples": len(_ROWS),
            "split_policy": {
                "unit": "source video sequence",
                "validation_augmented": False,
                "validation_source_sequences": ["a", "b"],
            },
        },
        "results": {
            "best_validation": metrics,
            "best_validation_filtered": metrics,
            "raw_validation_posthoc_diagnostic": metrics,
        },
        "artifacts": {
            "validation_predictions_filtered": artifact,
            "validation_predictions_raw": artifact,
            "included_validation": included_validation_artifact,
        },
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_analysis_computes_bins_calibration_macro_and_gap_safe_temporal_metrics(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture(tmp_path)

    result = analyze_student_validation.analyze_validation(
        manifest_path,
        target_bin_edges_m=(0.20, 0.30),
        expected_run_manifest_sha256=sha256_file(manifest_path),
    )

    assert result["overall"]["count"] == 5
    assert set(result["per_sequence"]) == {"a", "b"}
    assert result["per_sequence"]["a"]["regression"]["count"] == 3
    assert result["macro_average"]["sequence_count"] == 2
    assert [value["count"] for value in result["target_bins"]] == [1, 2, 2]
    assert result["calibration"]["equation"].startswith("prediction_m = slope")
    assert result["calibration"]["predictions_modified"] is False
    assert result["calibration"]["slope"] == pytest.approx(0.9982456140350877)
    assert result["calibration"]["intercept_m"] == pytest.approx(0.0024561403508772117)
    temporal = result["temporal"]
    assert temporal["aggregate"]["consecutive_pair_count"] == 2
    assert temporal["aggregate"]["contiguous_segment_count"] == 3
    assert temporal["aggregate"]["excluded_nonconsecutive_adjacent_pair_count"] == 1
    assert temporal["per_sequence"]["a"]["consecutive_pair_count"] == 1
    assert temporal["per_sequence"]["a"]["excluded_nonconsecutive_adjacent_pair_count"] == 1
    assert temporal["per_sequence"]["b"]["consecutive_pair_count"] == 1
    assert temporal["aggregate"]["delta_error"]["mae_m"] == pytest.approx(0.02)
    assert temporal["aggregate"]["delta_error"]["bias_m"] == pytest.approx(-0.01)
    assert result["validation_checks"]["prediction_sample_ids_verified"] is True
    assert result["validation_checks"]["run_manifest_sha256_verified_against_expected"] is True


def test_atomic_json_output_replaces_target_without_leaving_temporary_file(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    result = analyze_student_validation.analyze_validation(manifest_path)
    output_path = tmp_path / "analysis.json"
    output_path.write_text("stale", encoding="utf-8")

    analyze_student_validation._atomic_write_json(output_path, result)

    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    assert list(tmp_path.glob(".analysis.json.*.tmp")) == []


def test_analysis_rejects_prediction_artifact_hash_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    prediction_path = tmp_path / "validation_predictions_filtered.csv"
    with prediction_path.open("a", encoding="utf-8") as target:
        target.write("\n")

    with pytest.raises(ValueError, match="prediction artifact SHA-256 mismatch"):
        analyze_student_validation.analyze_validation(manifest_path)


def test_analysis_rejects_run_manifest_metric_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["best_validation_filtered"]["mae_m"] += 0.1
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run manifest metric mismatch"):
        analyze_student_validation.analyze_validation(manifest_path)


def test_analysis_rejects_prediction_row_count_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["selected_validation_samples"] = len(_ROWS) + 1
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prediction row count mismatch"):
        analyze_student_validation.analyze_validation(manifest_path)


def test_analysis_rejects_prediction_ids_outside_hashed_validation_cohort(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture(tmp_path)
    included_path = tmp_path / "included_validation.txt"
    sample_ids = included_path.read_text(encoding="utf-8").splitlines()
    sample_ids[0] = "train:000000:hand0:aug=identity"
    included_path.write_text(
        "".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["included_validation"]["sha256"] = sha256_file(included_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample IDs differ"):
        analyze_student_validation.analyze_validation(manifest_path)


def test_analysis_rejects_frame_before_chronological_validation_tail(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["split_policy"].update(
        {
            "unit": "source video chronological frame tail",
            "per_sequence": {
                "a": {
                    "validation_start_frame_inclusive": 1,
                    "source_frames_total": 4,
                },
                "b": {
                    "validation_start_frame_inclusive": 4,
                    "source_frames_total": 6,
                },
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside chronological validation tail"):
        analyze_student_validation.analyze_validation(manifest_path)


def test_analysis_payload_is_stable_when_run_directory_moves(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_manifest = _write_fixture(first_dir)
    second_manifest = _write_fixture(second_dir)

    first = analyze_student_validation.analyze_validation(first_manifest)
    second = analyze_student_validation.analyze_validation(second_manifest)

    assert sha256_file(first_manifest) == sha256_file(second_manifest)
    assert first == second


def test_analysis_rejects_inconsistent_stored_error_column(tmp_path: Path) -> None:
    manifest_path = _write_fixture(tmp_path, corrupt_error=True)

    with pytest.raises(ValueError, match="stored prediction error is inconsistent"):
        analyze_student_validation.analyze_validation(manifest_path)


@pytest.mark.parametrize("value", ["", "0.2,0.2", "0.3,0.2", "0.2,nan"])
def test_parse_bin_edges_rejects_empty_nonfinite_or_nonincreasing_values(value: str) -> None:
    with pytest.raises((ValueError, pytest.UsageError)):
        analyze_student_validation._parse_bin_edges(value)
