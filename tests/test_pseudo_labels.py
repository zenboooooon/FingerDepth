import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fingertip_depth.artifacts import write_json
from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.pseudo_labels import (
    DEPTH_PRO_TEACHER_CONDITION_ID,
    generate_depth_pro_pseudo_labels,
)
from fingertip_depth.video_cache import (
    pixel_hash_sequence_sha256,
    pixel_sha256,
    sha256_file,
)


class _FakeDepthPro:
    def __init__(self) -> None:
        self.calls: list[tuple[CameraIntrinsics | None, str]] = []

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model": "fake Depth Pro",
            "checkpoint_sha256": "a" * 64,
        }

    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> SimpleNamespace:
        self.calls.append((intrinsics, camera_mode))
        depth = np.full(rgb.shape[:2], 0.25, dtype=np.float32)
        depth[1, 2] = 0.30
        return SimpleNamespace(
            depth_m=depth,
            inference_ms=4.0,
            device="test",
            extras={"output_focal_px": intrinsics.fx_px if intrinsics else None},
        )


def _landmark(index: int, *, u_px: int, v_px: int) -> dict[str, object]:
    names = {
        5: "INDEX_FINGER_MCP",
        6: "INDEX_FINGER_PIP",
        7: "INDEX_FINGER_DIP",
        8: "INDEX_FINGER_TIP",
    }
    return {
        "landmark_index": index,
        "landmark_name": names[index],
        "x_normalized": u_px / 5,
        "y_normalized": v_px / 4,
        "z_mediapipe_relative": -0.01 * index,
        "u_px": u_px,
        "v_px": v_px,
        "in_frame": True,
    }


def _prepared_fixture(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    frames = prepared / "frames"
    frames.mkdir(parents=True)
    rows = []
    for frame_index, value in enumerate((40, 50)):
        bgr = np.full((4, 5, 3), value, dtype=np.uint8)
        image_path = frames / f"frame_{frame_index:06d}.png"
        assert cv2.imwrite(str(image_path), bgr)
        base = {
            "frame_index": frame_index,
            "timestamp_ms": frame_index * 33,
            "image_path": f"frames/{image_path.name}",
            "width": 5,
            "height": 4,
            "png_sha256": sha256_file(image_path),
            "bgr_pixel_sha256": pixel_sha256(bgr),
            "camera_intrinsics": {
                "fx_px": 100.0,
                "fy_px": 100.0,
                "cx_px": 2.0,
                "cy_px": 1.5,
            },
        }
        if frame_index == 0:
            features = [_landmark(index, u_px=2, v_px=1) for index in (5, 6, 7, 8)]
            base.update(
                {
                    "hand_detected": True,
                    "status": "accepted",
                    "rejection_reasons": [],
                    "hands": [
                        {
                            "hand_index": 0,
                            "handedness": "Right",
                            "handedness_score": 0.95,
                            "landmarks": features,
                        }
                    ],
                    "selected_hand_index": 0,
                    "target_landmark": features[-1],
                }
            )
        else:
            base.update(
                {
                    "hand_detected": False,
                    "status": "rejected",
                    "rejection_reasons": ["no_hand"],
                    "hands": [],
                    "selected_hand_index": None,
                    "target_landmark": None,
                }
            )
        rows.append(base)

    records_path = prepared / "frames.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "format": "fingertip-depth-pseudo-label-inputs",
        "format_version": 1,
        "source": {
            "mode": "test",
            "video_sha256": "b" * 64,
            "frame_cache_manifest_sha256": "c" * 64,
        },
        "frames_jsonl": {
            "relative_path": "frames.jsonl",
            "sha256": sha256_file(records_path),
        },
        "frames": {
            "relative_directory": "frames",
            "pixel_sequence_sha256": pixel_hash_sequence_sha256(
                str(row["bgr_pixel_sha256"]) for row in rows
            ),
            "pixel_sequence_encoding": (
                "sha256 of concatenated lowercase hexadecimal BGR pixel SHA-256 "
                "digests encoded as ASCII"
            ),
        },
        "frame_count": 2,
        "accepted_frame_count": 1,
        "rejected_frame_count": 1,
        "feature_landmarks": {
            "indices": [5, 6, 7, 8],
            "names": [
                "INDEX_FINGER_MCP",
                "INDEX_FINGER_PIP",
                "INDEX_FINGER_DIP",
                "INDEX_FINGER_TIP",
            ],
        },
        "target_landmark": {
            "index": 8,
            "name": "INDEX_FINGER_TIP",
            "depth_sampling": "single_pixel",
        },
        "camera": {
            "focal_35mm_equivalent_mm": 36.0,
            "image_size_px": {"width": 5, "height": 4},
            "intrinsics": rows[0]["camera_intrinsics"],
        },
    }
    manifest_path = prepared / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _selection_report(path: Path, prepared: Path) -> None:
    manifest = json.loads(prepared.read_text(encoding="utf-8"))
    write_json(
        path,
        {
            "expected_range_m": {"min": 0.2, "max": 0.3},
            "audit": {
                "source": {"sha256": manifest["source"]["video_sha256"]},
                "frame_cache": {
                    "manifest_sha256": manifest["source"]["frame_cache_manifest_sha256"]
                },
                "input_pixels": {
                    "sequence_sha256": manifest["frames"]["pixel_sequence_sha256"]
                },
            },
            "conditions": [
                {
                    "id": DEPTH_PRO_TEACHER_CONDITION_ID,
                    "band": {"band_violation_m": {"mean": 0.029}},
                }
            ],
        },
    )


def test_generates_single_pixel_labels_xyz_and_sequence_level_split(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    report = tmp_path / "selection.json"
    _selection_report(report, prepared)
    estimator = _FakeDepthPro()
    output = tmp_path / "dataset"

    manifest = generate_depth_pro_pseudo_labels(
        prepared_manifest_path=prepared,
        output_dir=output,
        estimator=estimator,
        sequence_id="recording_01",
        split="train",
        frame_transfer_mode="copy",
        expected_prepared_manifest_sha256=sha256_file(prepared),
        teacher_selection_report_path=report,
    )

    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 1
    sample = samples[0]
    assert [item["landmark_index"] for item in sample["hand"]["feature_landmarks"]] == [
        5,
        6,
        7,
        8,
    ]
    assert sample["target"]["z_teacher_m"] == pytest.approx(0.3)
    assert sample["target"]["camera_xyz_m"] == pytest.approx(
        {"x_m": 0.0, "y_m": -0.0015, "z_m": 0.3}
    )
    assert sample["target"]["depth_sampling"] == "single_pixel"
    assert sample["split"] == "train"
    assert estimator.calls and estimator.calls[0][1] == "approx_focal"
    assert estimator.calls[0][0] is not None
    assert manifest["ground_truth"] is False
    assert manifest["teacher"]["condition_id"] == DEPTH_PRO_TEACHER_CONDITION_ID
    assert manifest["teacher"]["selection_evidence"]["value"] == pytest.approx(0.029)
    assert manifest["teacher"]["selection_evidence"]["provenance_verified"] is True
    assert manifest["counts"]["accepted_samples"] == 1
    assert manifest["counts"]["rejected_frames_considered"] == 1
    assert manifest["counts"]["source_frames_considered"] == 2
    prepared_rows = [
        json.loads(line)
        for line in (prepared.parent / "frames.jsonl").read_text().splitlines()
    ]
    assert manifest["prepared_inputs"]["pixel_hash_sequence_sha256"] == (
        pixel_hash_sequence_sha256(
            str(row["bgr_pixel_sha256"]) for row in prepared_rows
        )
    )
    assert (output / "trajectories" / "recording_01.csv").is_file()
    assert (output / "trajectories" / "recording_01.ply").is_file()
    assert (output / "visualizations" / "recording_01_views.png").is_file()
    with (output / "targets.csv").open(encoding="utf-8", newline="") as source:
        target_rows = list(csv.DictReader(source))
    assert len(target_rows) == 1
    assert target_rows[0]["sample_id"] == "recording_01:000000:hand0"
    assert float(target_rows[0]["y_m"]) == pytest.approx(-0.0015)
    assert float(target_rows[0]["z_teacher_m"]) == pytest.approx(0.3)
    assert manifest["artifacts"]["targets_csv"]["sha256"] == sha256_file(
        output / "targets.csv"
    )
    assert (output / "splits" / "train.txt").read_text() == (
        "recording_01:000000:hand0\n"
    )


def test_rejects_tampered_prepared_frames_jsonl(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    records = prepared.parent / "frames.jsonl"
    records.write_text(records.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frames.jsonl SHA-256 mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=tmp_path / "dataset",
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
        )


def test_rejects_unexpected_prepared_manifest_hash(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)

    with pytest.raises(ValueError, match="prepared manifest SHA-256 mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=tmp_path / "dataset",
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
            expected_prepared_manifest_sha256="0" * 64,
        )


def test_rejects_mismatched_prepared_pixel_hash_sequence(tmp_path: Path) -> None:
    prepared = _prepared_fixture(tmp_path)
    manifest = json.loads(prepared.read_text(encoding="utf-8"))
    manifest["frames"]["pixel_sequence_sha256"] = "0" * 64
    write_json(prepared, manifest)

    with pytest.raises(ValueError, match="pixel hash sequence SHA-256 mismatch"):
        generate_depth_pro_pseudo_labels(
            prepared_manifest_path=prepared,
            output_dir=tmp_path / "dataset",
            estimator=_FakeDepthPro(),
            sequence_id="recording_01",
        )
