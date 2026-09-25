"""Audit and analyze a Phase 8 student validation-prediction artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from fingertip_depth.student_training import (
    TRAINING_RUN_FORMAT,
    TRAINING_RUN_FORMAT_VERSION,
    regression_metrics,
)
from fingertip_depth.video_cache import sha256_file

ANALYSIS_FORMAT = "fingertip-depth-student-validation-analysis"
ANALYSIS_FORMAT_VERSION = 1
DEFAULT_TARGET_BIN_EDGES_M = (0.22, 0.28, 0.34, 0.40)
_CSV_FIELDS = (
    "sample_id",
    "source_sequence_id",
    "frame_index",
    "augmentation_variant",
    "target_m",
    "prediction_m",
    "error_m",
    "absolute_error_m",
)
_METRIC_RELATIVE_TOLERANCE = 1e-9
_METRIC_ABSOLUTE_TOLERANCE = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    return digest


def _contained_file(root: Path, relative_path: object) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise ValueError(f"artifact path must be relative: {relative}")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"artifact path escapes the run directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parse_bin_edges(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        try:
            edges = tuple(float(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "target bin edges must be comma-separated numbers"
            ) from error
    else:
        edges = tuple(float(item) for item in value)
    if not edges:
        raise ValueError("at least one target bin edge is required")
    if any(not math.isfinite(edge) for edge in edges):
        raise ValueError("target bin edges must be finite")
    if any(second <= first for first, second in pairwise(edges)):
        raise ValueError("target bin edges must be strictly increasing")
    return edges


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    sequence_frames: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
            raise ValueError(
                f"unexpected prediction CSV columns: expected {_CSV_FIELDS}, got {reader.fieldnames}"
            )
        for line_number, raw in enumerate(reader, start=2):
            sample_id = str(raw["sample_id"])
            sequence_id = str(raw["source_sequence_id"])
            augmentation = str(raw["augmentation_variant"])
            if not sample_id or not sequence_id:
                raise ValueError(f"empty sample or sequence ID at {path}:{line_number}")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id at {path}:{line_number}: {sample_id}")
            if augmentation != "identity":
                raise ValueError(
                    f"validation prediction must be unaugmented identity data: {path}:{line_number}"
                )
            try:
                frame_index = int(raw["frame_index"])
                target_m = float(raw["target_m"])
                prediction_m = float(raw["prediction_m"])
                stored_error_m = float(raw["error_m"])
                stored_absolute_error_m = float(raw["absolute_error_m"])
            except ValueError as error:
                raise ValueError(f"invalid numeric field at {path}:{line_number}") from error
            values = (target_m, prediction_m, stored_error_m, stored_absolute_error_m)
            if frame_index < 0 or not all(math.isfinite(item) for item in values):
                raise ValueError(f"invalid finite prediction row at {path}:{line_number}")
            if target_m <= 0.0:
                raise ValueError(f"target depth must be positive at {path}:{line_number}")
            error_m = prediction_m - target_m
            if not math.isclose(
                stored_error_m,
                error_m,
                rel_tol=_METRIC_RELATIVE_TOLERANCE,
                abs_tol=_METRIC_ABSOLUTE_TOLERANCE,
            ) or not math.isclose(
                stored_absolute_error_m,
                abs(error_m),
                rel_tol=_METRIC_RELATIVE_TOLERANCE,
                abs_tol=_METRIC_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(f"stored prediction error is inconsistent at {path}:{line_number}")
            frame_key = (sequence_id, frame_index)
            if frame_key in sequence_frames:
                raise ValueError(
                    "duplicate source_sequence_id/frame_index in validation predictions: "
                    f"{frame_key}"
                )
            sample_ids.add(sample_id)
            sequence_frames.add(frame_key)
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "augmentation_variant": augmentation,
                    "target_m": target_m,
                    "prediction_m": prediction_m,
                }
            )
    if not rows:
        raise ValueError(f"prediction CSV is empty: {path}")
    return rows


def _read_sample_ids(path: Path) -> list[str]:
    sample_ids = path.read_text(encoding="utf-8").splitlines()
    if not sample_ids or any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"sample-ID artifact must contain non-empty lines: {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"sample-ID artifact contains duplicates: {path}")
    return sample_ids


def _verified_artifact(
    root: Path,
    entry: object,
    *,
    field: str,
) -> tuple[Path, str, str]:
    if not isinstance(entry, Mapping):
        raise TypeError(f"{field} must be an object")
    relative_path = str(entry.get("relative_path"))
    path = _contained_file(root, relative_path)
    expected_sha256 = _require_sha256(entry.get("sha256"), field=f"{field} SHA-256")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{field} SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}"
        )
    return path, observed_sha256, relative_path


def _expected_validation_sample_ids(
    *,
    cohort: str,
    run_root: Path,
    artifacts: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    artifact_key = "included_validation" if cohort == "filtered" else "identity_decisions"
    path, digest, relative_path = _verified_artifact(
        run_root,
        artifacts.get(artifact_key),
        field=f"run artifact {artifact_key}",
    )
    if cohort == "filtered":
        sample_ids = _read_sample_ids(path)
    else:
        sample_ids = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError(f"identity decision {line_number} must be an object")
                if value.get("split") == "validation":
                    sample_ids.append(str(value.get("identity_sample_id", "")))
        if not sample_ids or any(not sample_id for sample_id in sample_ids):
            raise ValueError("identity decisions contain no valid raw-validation sample IDs")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("identity decisions repeat a raw-validation sample ID")
    return sample_ids, {
        "artifact_key": artifact_key,
        "relative_path": relative_path,
        "sha256": digest,
    }


def _validate_chronological_boundaries(
    rows: Sequence[Mapping[str, Any]], split_policy: Mapping[str, Any]
) -> bool:
    if split_policy.get("unit") != "source video chronological frame tail":
        return False
    per_sequence = split_policy.get("per_sequence")
    if not isinstance(per_sequence, Mapping):
        raise TypeError("chronological split per_sequence metadata must be an object")
    for row in rows:
        sequence_id = str(row["source_sequence_id"])
        boundary = per_sequence.get(sequence_id)
        if not isinstance(boundary, Mapping):
            raise TypeError(f"missing chronological validation boundary: {sequence_id}")
        validation_start = int(boundary.get("validation_start_frame_inclusive", -1))
        source_frames_total = int(boundary.get("source_frames_total", -1))
        frame_index = int(row["frame_index"])
        if not 0 <= validation_start < source_frames_total:
            raise ValueError(f"invalid chronological validation boundary: {sequence_id}")
        if not validation_start <= frame_index < source_frames_total:
            raise ValueError(
                f"prediction frame lies outside chronological validation tail: "
                f"{sequence_id} frame {frame_index}"
            )
    return True


def _legacy_temporal_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    absolute_errors: list[float] = []
    by_sequence: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(str(row["source_sequence_id"]), []).append(row)
    for sequence_rows in by_sequence.values():
        ordered = sorted(sequence_rows, key=lambda row: int(row["frame_index"]))
        for previous, current in pairwise(ordered):
            if int(current["frame_index"]) != int(previous["frame_index"]) + 1:
                continue
            prediction_delta = float(current["prediction_m"]) - float(previous["prediction_m"])
            target_delta = float(current["target_m"]) - float(previous["target_m"])
            absolute_errors.append(abs(prediction_delta - target_delta))
    if not absolute_errors:
        return {"consecutive_pair_count": 0, "delta_mae_m": None, "delta_mae_cm": None}
    delta_mae_m = float(np.mean(np.asarray(absolute_errors, dtype=np.float64)))
    return {
        "consecutive_pair_count": len(absolute_errors),
        "delta_mae_m": delta_mae_m,
        "delta_mae_cm": 100.0 * delta_mae_m,
    }


def _assert_metric_value(actual: object, expected: object, *, field: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(
                f"run manifest metric mismatch for {field}: {expected!r} != {actual!r}"
            )
        return
    if isinstance(actual, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        if int(actual) != int(expected):
            raise ValueError(f"run manifest metric mismatch for {field}: {expected} != {actual}")
        return
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError) as error:
        raise ValueError(f"run manifest metric is not numeric for {field}") from error
    if not math.isclose(
        actual_float,
        expected_float,
        rel_tol=_METRIC_RELATIVE_TOLERANCE,
        abs_tol=_METRIC_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(
            f"run manifest metric mismatch for {field}: {expected_float} != {actual_float}"
        )


def _validate_overall_metrics(actual: Mapping[str, Any], expected: object, *, field: str) -> None:
    if not isinstance(expected, Mapping):
        raise TypeError(f"run manifest {field} must be an object")
    for key, actual_value in actual.items():
        if key not in expected:
            raise ValueError(f"run manifest {field} is missing metric {key}")
        _assert_metric_value(actual_value, expected[key], field=f"{field}.{key}")


def _series_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "mean_m": float(np.mean(values)),
        "std_m": float(np.std(values)),
        "range_m": float(np.max(values) - np.min(values)),
    }


def _calibration_diagnostics(target_m: np.ndarray, prediction_m: np.ndarray) -> dict[str, Any]:
    target_summary = _series_summary(target_m)
    prediction_summary = _series_summary(prediction_m)
    target_centered = target_m - float(np.mean(target_m))
    prediction_centered = prediction_m - float(np.mean(prediction_m))
    target_sum_squares = float(np.dot(target_centered, target_centered))
    prediction_sum_squares = float(np.dot(prediction_centered, prediction_centered))
    slope: float | None = None
    intercept_m: float | None = None
    r_squared: float | None = None
    if target_m.size >= 2 and target_sum_squares > 0.0:
        slope = float(np.dot(target_centered, prediction_centered) / target_sum_squares)
        intercept_m = float(np.mean(prediction_m) - slope * np.mean(target_m))
        if prediction_sum_squares > 0.0:
            fitted = slope * target_m + intercept_m
            residual_sum_squares = float(np.sum(np.square(prediction_m - fitted)))
            r_squared = float(1.0 - residual_sum_squares / prediction_sum_squares)
    target_std = target_summary["std_m"]
    target_range = target_summary["range_m"]
    return {
        "equation": "prediction_m = slope * target_m + intercept_m",
        "diagnostic_only": True,
        "predictions_modified": False,
        "slope": slope,
        "intercept_m": intercept_m,
        "r_squared": r_squared,
        "target": target_summary,
        "prediction": prediction_summary,
        "prediction_to_target_std_ratio": (
            prediction_summary["std_m"] / target_std if target_std > 0.0 else None
        ),
        "prediction_to_target_range_ratio": (
            prediction_summary["range_m"] / target_range if target_range > 0.0 else None
        ),
    }


def _delta_error_metrics(
    prediction_delta_m: Sequence[float], target_delta_m: Sequence[float]
) -> dict[str, Any]:
    if not prediction_delta_m:
        return {
            "count": 0,
            "mse_m2": None,
            "rmse_m": None,
            "rmse_cm": None,
            "mae_m": None,
            "mae_cm": None,
            "median_absolute_error_m": None,
            "median_absolute_error_cm": None,
            "p95_absolute_error_m": None,
            "p95_absolute_error_cm": None,
            "bias_m": None,
            "bias_cm": None,
            "max_absolute_error_m": None,
            "pearson_r": None,
        }
    metrics = regression_metrics(
        np.asarray(prediction_delta_m, dtype=np.float64),
        np.asarray(target_delta_m, dtype=np.float64),
    )
    metrics.pop("negative_prediction_count")
    return metrics


def _temporal_sequence_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    prediction_deltas: list[float] = []
    target_deltas: list[float] = []
    excluded_gaps = 0
    for previous, current in pairwise(ordered):
        if int(current["frame_index"]) != int(previous["frame_index"]) + 1:
            excluded_gaps += 1
            continue
        prediction_deltas.append(float(current["prediction_m"]) - float(previous["prediction_m"]))
        target_deltas.append(float(current["target_m"]) - float(previous["target_m"]))
    return {
        "observation_count": len(ordered),
        "contiguous_segment_count": 1 + excluded_gaps,
        "excluded_nonconsecutive_adjacent_pair_count": excluded_gaps,
        "consecutive_pair_count": len(prediction_deltas),
        "delta_error": _delta_error_metrics(prediction_deltas, target_deltas),
        "_prediction_deltas": prediction_deltas,
        "_target_deltas": target_deltas,
    }


def _temporal_metrics(
    rows_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    per_sequence: dict[str, dict[str, Any]] = {}
    all_prediction_deltas: list[float] = []
    all_target_deltas: list[float] = []
    total_segments = 0
    total_excluded_gaps = 0
    for sequence_id in sorted(rows_by_sequence):
        result = _temporal_sequence_metrics(rows_by_sequence[sequence_id])
        all_prediction_deltas.extend(result.pop("_prediction_deltas"))
        all_target_deltas.extend(result.pop("_target_deltas"))
        total_segments += int(result["contiguous_segment_count"])
        total_excluded_gaps += int(result["excluded_nonconsecutive_adjacent_pair_count"])
        per_sequence[sequence_id] = result
    return {
        "definition": (
            "frame-to-frame deltas are evaluated only when frame_index increases by exactly one "
            "within a source sequence; gaps and sequence boundaries are never crossed"
        ),
        "aggregate": {
            "source_sequence_count": len(per_sequence),
            "contiguous_segment_count": total_segments,
            "excluded_nonconsecutive_adjacent_pair_count": total_excluded_gaps,
            "consecutive_pair_count": len(all_prediction_deltas),
            "delta_error": _delta_error_metrics(all_prediction_deltas, all_target_deltas),
        },
        "per_sequence": per_sequence,
    }


def _macro_metrics(per_sequence: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "mse_m2",
        "rmse_m",
        "rmse_cm",
        "mae_m",
        "mae_cm",
        "median_absolute_error_m",
        "median_absolute_error_cm",
        "p95_absolute_error_m",
        "p95_absolute_error_cm",
        "bias_m",
        "bias_cm",
        "max_absolute_error_m",
    )
    result: dict[str, Any] = {
        "definition": "unweighted arithmetic mean of each source sequence's metric",
        "sequence_count": len(per_sequence),
    }
    for metric_name in metric_names:
        result[f"mean_sequence_{metric_name}"] = float(
            np.mean(
                np.asarray(
                    [float(value["regression"][metric_name]) for value in per_sequence.values()],
                    dtype=np.float64,
                )
            )
        )
    negative_rates = [
        float(value["regression"]["negative_prediction_count"]) / int(value["regression"]["count"])
        for value in per_sequence.values()
    ]
    correlations = [
        float(value["regression"]["pearson_r"])
        for value in per_sequence.values()
        if value["regression"]["pearson_r"] is not None
    ]
    result["mean_sequence_negative_prediction_rate"] = float(np.mean(negative_rates))
    result["pearson_contributing_sequence_count"] = len(correlations)
    result["mean_sequence_pearson_r"] = (
        float(np.mean(np.asarray(correlations, dtype=np.float64))) if correlations else None
    )
    return result


def _target_bin_label(lower: float | None, upper: float | None) -> str:
    if lower is None:
        return f"target_m < {upper:g}"
    if upper is None:
        return f"target_m >= {lower:g}"
    return f"{lower:g} <= target_m < {upper:g}"


def _target_bin_metrics(
    target_m: np.ndarray, prediction_m: np.ndarray, edges_m: Sequence[float]
) -> list[dict[str, Any]]:
    edges = np.asarray(edges_m, dtype=np.float64)
    assignments = np.digitize(target_m, edges, right=False)
    results: list[dict[str, Any]] = []
    for index in range(len(edges_m) + 1):
        lower = None if index == 0 else float(edges[index - 1])
        upper = None if index == len(edges_m) else float(edges[index])
        selected = assignments == index
        count = int(np.count_nonzero(selected))
        results.append(
            {
                "bin_index": index,
                "label": _target_bin_label(lower, upper),
                "lower_bound_m": lower,
                "lower_inclusive": lower is not None,
                "upper_bound_m": upper,
                "upper_inclusive": False,
                "count": count,
                "regression": (
                    regression_metrics(prediction_m[selected], target_m[selected])
                    if count > 0
                    else None
                ),
            }
        )
    if sum(int(result["count"]) for result in results) != int(target_m.size):
        raise AssertionError("target bin counts do not cover every validation row exactly once")
    return results


def _atomic_write_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def analyze_validation(
    run_manifest_path: Path,
    *,
    cohort: str = "filtered",
    target_bin_edges_m: Sequence[float] = DEFAULT_TARGET_BIN_EDGES_M,
    expected_run_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one run and calculate deterministic post-training diagnostics."""

    edges_m = _parse_bin_edges(target_bin_edges_m)
    run_manifest_path = run_manifest_path.resolve()
    observed_manifest_sha256 = sha256_file(run_manifest_path)
    if expected_run_manifest_sha256 is not None:
        expected_digest = _require_sha256(
            expected_run_manifest_sha256,
            field="expected run manifest SHA-256",
        )
        if observed_manifest_sha256 != expected_digest:
            raise ValueError(
                "run manifest SHA-256 mismatch: "
                f"expected {expected_digest}, observed {observed_manifest_sha256}"
            )
    manifest = _read_json(run_manifest_path)
    if manifest.get("format") != TRAINING_RUN_FORMAT:
        raise ValueError("unsupported student training run format")
    if manifest.get("format_version") != TRAINING_RUN_FORMAT_VERSION:
        raise ValueError("unsupported student training run format version")
    task = manifest.get("task")
    if not isinstance(task, Mapping) or task.get("ground_truth") is not False:
        raise ValueError("Phase 8 validation must remain explicitly marked as pseudo-label data")
    cohort_fields = {
        "filtered": (
            "validation_predictions_filtered",
            "selected_validation_samples",
            "best_validation_filtered",
        ),
        "raw": (
            "validation_predictions_raw",
            "raw_validation_samples",
            "raw_validation_posthoc_diagnostic",
        ),
    }
    if cohort not in cohort_fields:
        raise ValueError(f"cohort must be one of {sorted(cohort_fields)}, got {cohort!r}")
    artifact_key, count_key, result_key = cohort_fields[cohort]
    artifacts = manifest.get("artifacts")
    dataset = manifest.get("dataset")
    results = manifest.get("results")
    if not isinstance(artifacts, Mapping):
        raise TypeError("run manifest artifacts must be an object")
    if not isinstance(dataset, Mapping):
        raise TypeError("run manifest dataset must be an object")
    if not isinstance(results, Mapping):
        raise TypeError("run manifest results must be an object")
    prediction_path, observed_prediction_sha256, prediction_relative_path = _verified_artifact(
        run_manifest_path.parent,
        artifacts.get(artifact_key),
        field="prediction artifact",
    )
    rows = _read_prediction_rows(prediction_path)
    expected_row_count = int(dataset.get(count_key, -1))
    if expected_row_count <= 0 or len(rows) != expected_row_count:
        raise ValueError(
            f"prediction row count mismatch: manifest {count_key}={expected_row_count}, "
            f"CSV rows={len(rows)}"
        )
    expected_sample_ids, sample_id_provenance = _expected_validation_sample_ids(
        cohort=cohort,
        run_root=run_manifest_path.parent,
        artifacts=artifacts,
    )
    observed_sample_ids = [str(row["sample_id"]) for row in rows]
    if len(observed_sample_ids) != len(expected_sample_ids) or set(observed_sample_ids) != set(
        expected_sample_ids
    ):
        raise ValueError("prediction sample IDs differ from the audited validation cohort artifact")
    split_policy = dataset.get("split_policy")
    if (
        not isinstance(split_policy, Mapping)
        or split_policy.get("validation_augmented") is not False
    ):
        raise ValueError("run manifest must declare unaugmented validation data")
    validation_sequences = {
        str(value) for value in split_policy.get("validation_source_sequences", [])
    }
    observed_sequences = {str(row["source_sequence_id"]) for row in rows}
    if not validation_sequences or not observed_sequences.issubset(validation_sequences):
        raise ValueError("prediction sequences are inconsistent with the run manifest split policy")
    chronological_boundaries_verified = _validate_chronological_boundaries(rows, split_policy)

    target_m = np.asarray([float(row["target_m"]) for row in rows], dtype=np.float64)
    prediction_m = np.asarray([float(row["prediction_m"]) for row in rows], dtype=np.float64)
    overall = regression_metrics(prediction_m, target_m)
    expected_metrics = results.get(result_key)
    _validate_overall_metrics(overall, expected_metrics, field=f"results.{result_key}")
    legacy_temporal = _legacy_temporal_metrics(rows)
    if not isinstance(expected_metrics, Mapping):
        raise TypeError(f"run manifest results.{result_key} must be an object")
    expected_temporal = expected_metrics.get("consecutive_frame_delta")
    _validate_overall_metrics(
        legacy_temporal,
        expected_temporal,
        field=f"results.{result_key}.consecutive_frame_delta",
    )
    if cohort == "filtered" and "best_validation" in results:
        _validate_overall_metrics(
            overall, results["best_validation"], field="results.best_validation"
        )

    rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sequence.setdefault(str(row["source_sequence_id"]), []).append(row)
    per_sequence: dict[str, dict[str, Any]] = {}
    for sequence_id in sorted(rows_by_sequence):
        sequence_rows = rows_by_sequence[sequence_id]
        sequence_target = np.asarray(
            [float(row["target_m"]) for row in sequence_rows], dtype=np.float64
        )
        sequence_prediction = np.asarray(
            [float(row["prediction_m"]) for row in sequence_rows], dtype=np.float64
        )
        per_sequence[sequence_id] = {
            "regression": regression_metrics(sequence_prediction, sequence_target),
            "target": _series_summary(sequence_target),
            "prediction": _series_summary(sequence_prediction),
        }

    return {
        "format": ANALYSIS_FORMAT,
        "format_version": ANALYSIS_FORMAT_VERSION,
        "source": {
            "run_manifest": {
                "filename": run_manifest_path.name,
                "sha256": observed_manifest_sha256,
                "format": manifest["format"],
                "format_version": manifest["format_version"],
            },
            "prediction_artifact": {
                "cohort": cohort,
                "artifact_key": artifact_key,
                "relative_path": prediction_relative_path,
                "sha256": observed_prediction_sha256,
                "row_count": len(rows),
            },
            "validation_sample_ids": sample_id_provenance,
            "target_provenance": {
                "target": task.get("target"),
                "ground_truth": False,
            },
        },
        "validation_checks": {
            "run_manifest_format_verified": True,
            "run_manifest_sha256_verified_against_expected": (
                expected_run_manifest_sha256 is not None
            ),
            "prediction_artifact_sha256_verified": True,
            "prediction_row_count_verified": True,
            "prediction_sample_ids_verified": True,
            "stored_error_columns_verified": True,
            "chronological_tail_boundary_check": (
                "verified" if chronological_boundaries_verified else "not_applicable"
            ),
            "overall_run_manifest_metrics_verified": True,
            "metric_relative_tolerance": _METRIC_RELATIVE_TOLERANCE,
            "metric_absolute_tolerance": _METRIC_ABSOLUTE_TOLERANCE,
        },
        "parameters": {
            "target_bin_edges_m": list(edges_m),
            "target_bin_interval_convention": "lower-inclusive and upper-exclusive",
        },
        "overall": overall,
        "per_sequence": per_sequence,
        "macro_average": _macro_metrics(per_sequence),
        "target_bins": _target_bin_metrics(target_m, prediction_m, edges_m),
        "calibration": _calibration_diagnostics(target_m, prediction_m),
        "temporal": _temporal_metrics(rows_by_sequence),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("filtered", "raw"), default="filtered")
    parser.add_argument(
        "--expected-run-manifest-sha256",
        help="optional pinned SHA-256 for the run manifest",
    )
    parser.add_argument(
        "--target-bin-edges-m",
        type=_parse_bin_edges,
        default=DEFAULT_TARGET_BIN_EDGES_M,
        help="strictly increasing comma-separated target-depth edges (default: 0.22,0.28,0.34,0.40)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="default: <run directory>/validation_analysis_<cohort>.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_validation(
        args.run_manifest,
        cohort=args.cohort,
        target_bin_edges_m=args.target_bin_edges_m,
        expected_run_manifest_sha256=args.expected_run_manifest_sha256,
    )
    output_path = args.output
    if output_path is None:
        output_path = args.run_manifest.resolve().parent / f"validation_analysis_{args.cohort}.json"
    _atomic_write_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "sha256": sha256_file(output_path.resolve()),
                "cohort": args.cohort,
                "count": result["overall"]["count"],
                "rmse_cm": result["overall"]["rmse_cm"],
                "mae_cm": result["overall"]["mae_cm"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
