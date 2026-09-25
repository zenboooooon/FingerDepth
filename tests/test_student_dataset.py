import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from fingertip_depth.artifacts import write_json
from fingertip_depth.coordinates import normalized_to_pixel
from fingertip_depth.geometry import CAMERA_COORDINATE_CONVENTION
from fingertip_depth.pseudo_labels import (
    DEPTH_PRO_TEACHER_CONDITION_ID,
    PSEUDO_LABEL_DATASET_FORMAT,
    PSEUDO_LABEL_DATASET_FORMAT_VERSION,
)
from fingertip_depth.student_dataset import (
    STUDENT_DATASET_FORMAT,
    _ceil_fractional_count,
    build_student_dataset,
)
from fingertip_depth.video_cache import pixel_sha256, sha256_file


def test_ceil_fractional_count_avoids_binary_float_boundary_drift() -> None:
    assert _ceil_fractional_count(100, 0.07) == 7
    assert _ceil_fractional_count(267, 0.07) == 19
    assert _ceil_fractional_count(3206, 0.07) == 225


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_dataset(
    root: Path,
    *,
    sequence_id: str,
    split: str,
    color: int,
    selection_evidence: bool = False,
    video_sha256: str | None = None,
    width: int = 5,
    height: int = 4,
    fx: float = 100.0,
    fy: float = 120.0,
    cx: float = 1.25,
    cy: float = 1.5,
    x_normalized: float = 0.2,
    y_normalized: float = 0.25,
    evidence_video_sha256: str | None = None,
    filtering: dict[str, object] | None = None,
    source_commit: str = "1" * 40,
) -> Path:
    root.mkdir(parents=True)
    frames = root / "frames" / sequence_id
    frames.mkdir(parents=True)
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = color
    bgr[:, :, 1] = np.arange(width, dtype=np.uint8)
    bgr[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
    image_path = frames / "frame_000000.png"
    assert cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    u, v = normalized_to_pixel(
        x_normalized,
        y_normalized,
        width=width,
        height=height,
    )
    z = 0.3
    xyz = {
        "x_m": (u - cx) * z / fx,
        "y_m": (v - cy) * z / fy,
        "z_m": z,
    }
    landmark_names = {
        5: "INDEX_FINGER_MCP",
        6: "INDEX_FINGER_PIP",
        7: "INDEX_FINGER_DIP",
        8: "INDEX_FINGER_TIP",
    }
    features = [
        {
            "landmark_index": index,
            "landmark_name": landmark_names[index],
            "x_normalized": x_normalized,
            "y_normalized": y_normalized,
            "z_mediapipe_relative": -0.01 * index,
            "u_px": u,
            "v_px": v,
            "in_frame": True,
        }
        for index in (5, 6, 7, 8)
    ]
    base_id = f"{sequence_id}:000000:hand0"
    relative_image = image_path.relative_to(root).as_posix()
    sample = {
        "schema_version": 1,
        "sample_id": base_id,
        "sequence_id": sequence_id,
        "split": split,
        "frame_index": 0,
        "timestamp_ms": 0,
        "image": {
            "relative_path": relative_image,
            "width": width,
            "height": height,
            "png_sha256": sha256_file(image_path),
            "bgr_pixel_sha256": pixel_sha256(bgr),
        },
        "camera": {
            "fx_px": fx,
            "fy_px": fy,
            "cx_px": cx,
            "cy_px": cy,
            "model": "centered_pinhole_approximation",
            "intrinsics_source": "test approximation",
        },
        "hand": {
            "hand_index": 0,
            "handedness": "Right",
            "handedness_score": 0.9,
            "feature_landmarks": features,
        },
        "target": {
            "landmark_index": 8,
            "landmark_name": "INDEX_FINGER_TIP",
            "u_px": u,
            "v_px": v,
            "z_teacher_m": z,
            "camera_xyz_m": xyz,
            "depth_sampling": "single_pixel",
            "depth_valid": True,
        },
        "teacher": {
            "condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
            "camera_mode": "approx_focal",
            "input_focal_px": fx,
            "output_focal_px": fx,
            "confidence": None,
            "confidence_available": False,
            "depth_unit": "metre",
            "fitted_scale_or_offset_applied": False,
        },
    }
    samples_path = root / "samples.jsonl"
    samples_path.write_text(
        json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    targets_path = root / "targets.csv"
    with targets_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "sample_id",
                "sequence_id",
                "split",
                "frame_index",
                "timestamp_ms",
                "u_px",
                "v_px",
                "x_m",
                "y_m",
                "z_teacher_m",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": base_id,
                "sequence_id": sequence_id,
                "split": split,
                "frame_index": 0,
                "timestamp_ms": 0,
                "u_px": u,
                "v_px": v,
                "x_m": xyz["x_m"],
                "y_m": xyz["y_m"],
                "z_teacher_m": z,
            }
        )
    split_path = root / "splits" / f"{split}.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(f"{base_id}\n", encoding="utf-8")

    evidence = None
    if selection_evidence:
        evidence = {
            "report_sha256": _digest("selection-report"),
            "condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
            "metric": "approximate_range_band_violation_mean_m",
            "value": 0.029,
            "source_video_sha256": evidence_video_sha256
            or video_sha256
            or _digest(f"video-{sequence_id}"),
            "input_pixel_sequence_sha256": _digest(f"pixels-{sequence_id}"),
            "provenance_verified": True,
        }
    manifest = {
        "format": PSEUDO_LABEL_DATASET_FORMAT,
        "format_version": PSEUDO_LABEL_DATASET_FORMAT_VERSION,
        "label_type": "pseudo_label",
        "ground_truth": False,
        "pseudo_label_notice": "test pseudo labels are not ground truth",
        "sequence": {
            "id": sequence_id,
            "split": split,
            "split_policy": "explicit sequence-level split",
        },
        "prepared_inputs": {
            "manifest_sha256": _digest(f"prepared-{sequence_id}"),
            "frames_jsonl_sha256": _digest(f"frames-{sequence_id}"),
            "pixel_hash_sequence_sha256": _digest(f"pixels-{sequence_id}"),
            "source": {
                "video_sha256": video_sha256 or _digest(f"video-{sequence_id}"),
            },
        },
        "feature_landmarks": {
            "indices": [5, 6, 7, 8],
            "names": [landmark_names[index] for index in (5, 6, 7, 8)],
        },
        "target_landmark": {
            "index": 8,
            "name": "INDEX_FINGER_TIP",
            "depth_sampling": "single_pixel",
        },
        "teacher": {
            "condition_id": DEPTH_PRO_TEACHER_CONDITION_ID,
            "camera_mode": "approx_focal",
            "depth_sampling": "single_pixel",
            "metric_depth_unit": "metre",
            "depth_semantics_assumption": "optical-axis Z",
            "fitted_scale_or_offset_applied": False,
            "confidence_available": False,
            "model": {
                "backend": "depth_pro",
                "model": "fake Depth Pro",
                "source_repository": "https://example.test/depth-pro",
                "source_commit": source_commit,
                "checkpoint_repository": "example/DepthPro",
                "checkpoint_revision": "2" * 40,
                "checkpoint_filename": "depth_pro.pt",
                "checkpoint_sha256": "a" * 64,
                "precision": "float16",
                "depth_unit": "metre",
                "device": f"runtime-device-{sequence_id}",
                "checkpoint_path": str(root / "runtime" / "depth_pro.pt"),
            },
            "selection_evidence": evidence,
        },
        "camera_coordinate_system": {
            "convention": CAMERA_COORDINATE_CONVENTION,
            "x_formula": "(u_px - cx_px) * z_m / fx_px",
            "y_formula": "(v_px - cy_px) * z_m / fy_px",
            "z_formula": "teacher optical-axis depth in metres",
        },
        "filtering": filtering
        or {
            "depth_extraction": "single pixel only; Phase 3 ROI comparison skipped",
            "temporal_filtering": "none; Phase 4 temporal filtering skipped",
            "label_clipping": "none",
            "structural_checks_only": True,
        },
        "counts": {"accepted_samples": 1},
        "artifacts": {
            "samples_jsonl": {
                "relative_path": "samples.jsonl",
                "sha256": sha256_file(samples_path),
            },
            "targets_csv": {
                "relative_path": "targets.csv",
                "sha256": sha256_file(targets_path),
            },
            "split": {
                "relative_path": split_path.relative_to(root).as_posix(),
                "sha256": sha256_file(split_path),
            },
        },
    }
    manifest_path = root / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _three_sources(tmp_path: Path) -> list[Path]:
    return [
        _source_dataset(
            tmp_path / "source_2030",
            sequence_id="finger_movement_2030",
            split="train",
            color=30,
            selection_evidence=True,
        ),
        _source_dataset(
            tmp_path / "source_3",
            sequence_id="finger_movement_3",
            split="train",
            color=60,
        ),
        _source_dataset(
            tmp_path / "source_valid",
            sequence_id="finger_movement",
            split="validation",
            color=90,
        ),
    ]


def _expand_source_timeline(
    manifest_path: Path,
    *,
    frame_indices: tuple[int, ...],
    source_frames_total: int,
) -> Path:
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_path = root / manifest["artifacts"]["samples_jsonl"]["relative_path"]
    base = json.loads(samples_path.read_text(encoding="utf-8").splitlines()[0])
    base_image = cv2.imread(str(root / base["image"]["relative_path"]), cv2.IMREAD_COLOR)
    assert base_image is not None
    sequence_id = manifest["sequence"]["id"]
    split = manifest["sequence"]["split"]
    rows: list[dict[str, object]] = []
    for frame_index in frame_indices:
        bgr = base_image.copy()
        bgr[0, 0, 0] = (int(base_image[0, 0, 0]) + frame_index + 1) % 256
        image_path = root / "frames" / sequence_id / f"frame_{frame_index:06d}.png"
        assert cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        row = copy.deepcopy(base)
        sample_id = f"{sequence_id}:{frame_index:06d}:hand0"
        row.update(
            {
                "sample_id": sample_id,
                "frame_index": frame_index,
                "timestamp_ms": frame_index * 33,
            }
        )
        row["image"].update(
            {
                "relative_path": image_path.relative_to(root).as_posix(),
                "png_sha256": sha256_file(image_path),
                "bgr_pixel_sha256": pixel_sha256(bgr),
            }
        )
        rows.append(row)
    samples_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    targets_path = root / manifest["artifacts"]["targets_csv"]["relative_path"]
    with targets_path.open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(
            target_file,
            fieldnames=[
                "sample_id",
                "sequence_id",
                "split",
                "frame_index",
                "timestamp_ms",
                "u_px",
                "v_px",
                "x_m",
                "y_m",
                "z_teacher_m",
            ],
        )
        writer.writeheader()
        for row in rows:
            target = row["target"]
            xyz = target["camera_xyz_m"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "sequence_id": sequence_id,
                    "split": split,
                    "frame_index": row["frame_index"],
                    "timestamp_ms": row["timestamp_ms"],
                    "u_px": target["u_px"],
                    "v_px": target["v_px"],
                    "x_m": xyz["x_m"],
                    "y_m": xyz["y_m"],
                    "z_teacher_m": target["z_teacher_m"],
                }
            )
    split_path = root / manifest["artifacts"]["split"]["relative_path"]
    split_path.write_text(
        "".join(f"{row['sample_id']}\n" for row in rows),
        encoding="utf-8",
    )
    manifest["counts"] = {
        "source_frames_total": source_frames_total,
        "accepted_samples": len(rows),
    }
    manifest["prepared_inputs"]["source"]["source_frame_count"] = source_frames_total
    for name, path in (
        ("samples_jsonl", samples_path),
        ("targets_csv", targets_path),
        ("split", split_path),
    ):
        manifest["artifacts"][name]["sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)
    return manifest_path


def test_builds_chronological_tail_split_from_raw_video_frames(
    tmp_path: Path,
) -> None:
    manifests = [
        _expand_source_timeline(
            path,
            frame_indices=tuple(range(10)),
            source_frames_total=10,
        )
        for path in _three_sources(tmp_path)
    ]
    output = tmp_path / "student_tail"
    manifest = build_student_dataset(
        source_manifest_paths=manifests,
        output_dir=output,
        frame_transfer_mode="copy",
        validation_tail_fraction=0.2,
    )
    policy = manifest["split_policy"]
    assert policy["augmented_views_inherit_source_split"] is False
    assert policy["augmented_views_inherit_assigned_split"] is True

    assert policy["unit"] == "source video chronological frame tail"
    assert policy["validation_tail_fraction"] == 0.2
    assert policy["train_source_sequences"] == policy["validation_source_sequences"]
    assert set(policy["per_sequence"]) == {
        "finger_movement",
        "finger_movement_2030",
        "finger_movement_3",
    }
    for boundary in policy["per_sequence"].values():
        assert boundary == {
            "source_frames_total": 10,
            "validation_frame_count": 2,
            "validation_start_frame_inclusive": 8,
            "train_frame_end_exclusive": 8,
            "accepted_train_identity_samples": 8,
            "accepted_validation_identity_samples": 2,
            "first_accepted_validation_frame": 8,
            "last_accepted_validation_frame": 9,
        }
    assert manifest["counts"]["train_identity_samples"] == 24
    assert manifest["counts"]["train_hflip_samples"] == 24
    assert manifest["counts"]["train_samples_total"] == 48
    assert manifest["counts"]["validation_identity_samples"] == 6
    assert manifest["counts"]["validation_hflip_samples"] == 0
    assert manifest["counts"]["samples_total"] == 54

    samples = [
        json.loads(line)
        for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for sample in samples:
        frame_index = int(sample["frame_index"])
        if frame_index >= 8:
            assert sample["split"] == "validation"
            assert sample["augmentation"]["variant"] == "identity"
        else:
            assert sample["split"] == "train"
    train_source_ids = {
        sample["source_sample_id"] for sample in samples if sample["split"] == "train"
    }
    validation_source_ids = {
        sample["source_sample_id"] for sample in samples if sample["split"] == "validation"
    }
    assert train_source_ids.isdisjoint(validation_source_ids)


def test_chronological_tail_uses_raw_timeline_with_sparse_accepted_frames(
    tmp_path: Path,
) -> None:
    manifests = [
        _expand_source_timeline(
            path,
            frame_indices=(0, 1, 2, 8, 9),
            source_frames_total=10,
        )
        for path in _three_sources(tmp_path)
    ]
    output = tmp_path / "student_sparse_tail"
    manifest = build_student_dataset(
        source_manifest_paths=manifests,
        output_dir=output,
        frame_transfer_mode="copy",
        validation_tail_fraction=0.2,
    )

    for boundary in manifest["split_policy"]["per_sequence"].values():
        assert boundary["validation_start_frame_inclusive"] == 8
        assert boundary["accepted_train_identity_samples"] == 3
        assert boundary["accepted_validation_identity_samples"] == 2
    assert manifest["counts"]["train_identity_samples"] == 9
    assert manifest["counts"]["train_hflip_samples"] == 9
    assert manifest["counts"]["validation_identity_samples"] == 6


def test_chronological_tail_split_requires_source_frame_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_frames_total"):
        build_student_dataset(
            source_manifest_paths=_three_sources(tmp_path),
            output_dir=tmp_path / "student_tail",
            validation_tail_fraction=0.07,
        )


def test_rejects_conflicting_phase7_source_frame_counts(tmp_path: Path) -> None:
    source = _three_sources(tmp_path)[0]
    path = _expand_source_timeline(source, frame_indices=(0, 1), source_frames_total=10)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["prepared_inputs"]["source"]["source_frame_count"] = 11
    write_json(path, manifest)

    with pytest.raises(ValueError, match="source frame counts are inconsistent"):
        build_student_dataset(
            source_manifest_paths=[path],
            output_dir=tmp_path / "student_bad_counts",
        )


def test_builds_train_doubled_validation_unaugmented_with_exact_geometry(
    tmp_path: Path,
) -> None:
    manifests = _three_sources(tmp_path)
    output = tmp_path / "student"
    manifest = build_student_dataset(
        source_manifest_paths=list(reversed(manifests)),
        output_dir=output,
        frame_transfer_mode="copy",
        expected_source_manifest_sha256s=[sha256_file(path) for path in reversed(manifests)],
    )

    assert manifest["format"] == STUDENT_DATASET_FORMAT
    assert manifest["split_policy"]["train_source_sequences"] == [
        "finger_movement_2030",
        "finger_movement_3",
    ]
    assert manifest["split_policy"]["validation_source_sequences"] == ["finger_movement"]
    assert manifest["counts"] == {
        "source_sequences_total": 3,
        "train_source_sequences": 2,
        "validation_source_sequences": 1,
        "source_identity_samples_total": 3,
        "train_identity_samples": 2,
        "train_hflip_samples": 2,
        "train_samples_total": 4,
        "train_exactly_doubled": True,
        "validation_identity_samples": 1,
        "validation_hflip_samples": 0,
        "validation_samples_total": 1,
        "validation_unaugmented": True,
        "samples_total": 5,
        "materialized_image_files": 5,
    }
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    by_id = {sample["sample_id"]: sample for sample in samples}
    base_id = "finger_movement_2030:000000:hand0"
    identity = by_id[f"{base_id}:aug=identity"]
    hflip = by_id[f"{base_id}:aug=hflip"]
    assert hflip["source_sample_id"] == base_id
    assert hflip["view_sequence_id"] == "finger_movement_2030@hflip"
    assert hflip["target"]["u_px"] == 3
    assert hflip["target"]["v_px"] == identity["target"]["v_px"]
    assert hflip["target"]["z_teacher_m"] == identity["target"]["z_teacher_m"]
    assert hflip["target"]["camera_xyz_m"]["x_m"] == pytest.approx(
        -identity["target"]["camera_xyz_m"]["x_m"]
    )
    assert hflip["target"]["camera_xyz_m"]["y_m"] == identity["target"]["camera_xyz_m"]["y_m"]
    assert hflip["camera"]["cx_px"] == pytest.approx(2.75)
    landmark = hflip["hand"]["feature_landmarks"][0]
    assert landmark["x_normalized"] == pytest.approx(0.6)
    assert landmark["u_px"] == 3
    assert math.floor(landmark["x_normalized"] * 5 + 0.5) == 3
    assert hflip["hand"]["handedness_detected_source"] == "Right"
    assert hflip["hand"]["handedness_in_view"] == "Left"
    assert hflip["hand"]["handedness"] == "Left"
    assert hflip["teacher"]["label_reused_from_source"] is True
    assert hflip["teacher"]["teacher_inference_on_augmented_image"] is False

    identity_bgr = cv2.imread(str(output / identity["image"]["relative_path"]))
    hflip_bgr = cv2.imread(str(output / hflip["image"]["relative_path"]))
    assert identity_bgr is not None and hflip_bgr is not None
    assert np.array_equal(hflip_bgr, identity_bgr[:, ::-1])
    assert pixel_sha256(hflip_bgr) == hflip["image"]["bgr_pixel_sha256"]

    validation_ids = (output / "splits" / "validation.txt").read_text().splitlines()
    assert validation_ids == ["finger_movement:000000:hand0:aug=identity"]
    train_ids = (output / "splits" / "train.txt").read_text().splitlines()
    assert len(train_ids) == 4
    assert sum(identifier.endswith("aug=hflip") for identifier in train_ids) == 2
    scopes = manifest["teacher"]["selection_evidence_scopes"]
    assert [scope["source_sequence_id"] for scope in scopes] == ["finger_movement_2030"]
    for descriptor in (
        manifest["artifacts"]["samples_jsonl"],
        manifest["artifacts"]["targets_csv"],
        manifest["artifacts"]["train_split"],
        manifest["artifacts"]["validation_split"],
    ):
        assert sha256_file(output / descriptor["relative_path"]) == descriptor["sha256"]


def test_hardlinks_identity_but_materializes_hflip(tmp_path: Path) -> None:
    manifests = _three_sources(tmp_path)
    output = tmp_path / "student"
    build_student_dataset(
        source_manifest_paths=manifests,
        output_dir=output,
        frame_transfer_mode="hardlink",
    )
    source_image = tmp_path / "source_3" / "frames" / "finger_movement_3" / "frame_000000.png"
    identity_image = output / "frames" / "finger_movement_3" / "identity" / "frame_000000.png"
    hflip_image = output / "frames" / "finger_movement_3" / "hflip" / "frame_000000.png"
    assert source_image.stat().st_ino == identity_image.stat().st_ino
    assert source_image.stat().st_ino != hflip_image.stat().st_ino


def test_rejects_tampered_source_image_before_creating_output(tmp_path: Path) -> None:
    manifests = _three_sources(tmp_path)
    image = tmp_path / "source_3" / "frames" / "finger_movement_3" / "frame_000000.png"
    bgr = cv2.imread(str(image))
    assert bgr is not None
    bgr[0, 0] = 255
    assert cv2.imwrite(str(image), bgr)
    output = tmp_path / "student"

    with pytest.raises(ValueError, match="PNG SHA-256 mismatch"):
        build_student_dataset(source_manifest_paths=manifests, output_dir=output)
    assert not output.exists()


def test_rejects_duplicate_source_video_across_splits(tmp_path: Path) -> None:
    duplicate_video = "d" * 64
    train = _source_dataset(
        tmp_path / "train",
        sequence_id="train_sequence",
        split="train",
        color=10,
        video_sha256=duplicate_video,
    )
    validation = _source_dataset(
        tmp_path / "validation",
        sequence_id="validation_sequence",
        split="validation",
        color=20,
        video_sha256=duplicate_video,
    )
    with pytest.raises(ValueError, match="video SHA-256 values must be globally unique"):
        build_student_dataset(
            source_manifest_paths=[train, validation],
            output_dir=tmp_path / "student",
        )


def test_rejects_pinned_source_manifest_hash_before_output(tmp_path: Path) -> None:
    manifests = _three_sources(tmp_path)
    output = tmp_path / "student"
    expected = [sha256_file(path) for path in manifests]
    expected[1] = "0" * 64

    with pytest.raises(ValueError, match="source manifest SHA-256 mismatch before reading"):
        build_student_dataset(
            source_manifest_paths=manifests,
            expected_source_manifest_sha256s=expected,
            output_dir=output,
        )
    assert not output.exists()


def test_rejects_identity_pixels_shared_across_train_and_validation(tmp_path: Path) -> None:
    train = _source_dataset(
        tmp_path / "train",
        sequence_id="train_sequence",
        split="train",
        color=44,
    )
    validation = _source_dataset(
        tmp_path / "validation",
        sequence_id="validation_sequence",
        split="validation",
        color=44,
    )

    with pytest.raises(ValueError, match="BGR pixel hashes overlap"):
        build_student_dataset(
            source_manifest_paths=[train, validation],
            output_dir=tmp_path / "student",
        )


def test_rejects_selection_evidence_from_another_video(tmp_path: Path) -> None:
    train = _source_dataset(
        tmp_path / "train",
        sequence_id="train_sequence",
        split="train",
        color=11,
        selection_evidence=True,
        evidence_video_sha256="f" * 64,
    )
    validation = _source_dataset(
        tmp_path / "validation",
        sequence_id="validation_sequence",
        split="validation",
        color=12,
    )

    with pytest.raises(ValueError, match="selection evidence source video differs"):
        build_student_dataset(
            source_manifest_paths=[train, validation],
            output_dir=tmp_path / "student",
        )


def test_preserves_per_sample_resolution_and_reflects_off_center_k(tmp_path: Path) -> None:
    train = _source_dataset(
        tmp_path / "train",
        sequence_id="train_sequence",
        split="train",
        color=21,
        width=6,
        height=5,
        fx=80.0,
        fy=90.0,
        cx=0.7,
        cy=2.1,
        x_normalized=0.25,
        y_normalized=0.4,
    )
    validation = _source_dataset(
        tmp_path / "validation",
        sequence_id="validation_sequence",
        split="validation",
        color=22,
        width=7,
        height=6,
        fx=130.0,
        fy=140.0,
        cx=3.4,
        cy=2.2,
        x_normalized=0.3,
        y_normalized=0.5,
    )
    output = tmp_path / "student"
    build_student_dataset(
        source_manifest_paths=[train, validation],
        output_dir=output,
        frame_transfer_mode="copy",
    )
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    train_hflip = next(
        sample
        for sample in samples
        if sample["source_sequence_id"] == "train_sequence"
        and sample["augmentation"]["variant"] == "hflip"
    )
    validation_identity = next(
        sample for sample in samples if sample["source_sequence_id"] == "validation_sequence"
    )
    assert train_hflip["image"]["width"] == 6
    assert train_hflip["image"]["height"] == 5
    assert train_hflip["camera"]["fx_px"] == 80.0
    assert train_hflip["camera"]["fy_px"] == 90.0
    assert train_hflip["camera"]["cx_px"] == pytest.approx(4.3)
    landmark = train_hflip["hand"]["feature_landmarks"][0]
    assert normalized_to_pixel(
        landmark["x_normalized"],
        landmark["y_normalized"],
        width=6,
        height=5,
    ) == (landmark["u_px"], landmark["v_px"])
    assert validation_identity["image"]["width"] == 7
    assert validation_identity["camera"]["cx_px"] == 3.4


def test_rejects_filtering_and_stable_teacher_mismatches(tmp_path: Path) -> None:
    bad_filtering = {
        "depth_extraction": "single pixel only; Phase 3 ROI comparison skipped",
        "temporal_filtering": "moving average",
        "label_clipping": "none",
        "structural_checks_only": True,
    }
    filtered_train = _source_dataset(
        tmp_path / "filtered_train",
        sequence_id="filtered_train",
        split="train",
        color=31,
        filtering=bad_filtering,
    )
    validation = _source_dataset(
        tmp_path / "validation",
        sequence_id="validation",
        split="validation",
        color=32,
    )
    with pytest.raises(ValueError, match="single-pixel extraction"):
        build_student_dataset(
            source_manifest_paths=[filtered_train, validation],
            output_dir=tmp_path / "filtered_output",
        )

    teacher_train = _source_dataset(
        tmp_path / "teacher_train",
        sequence_id="teacher_train",
        split="train",
        color=33,
        source_commit="3" * 40,
    )
    with pytest.raises(ValueError, match="different teacher configurations"):
        build_student_dataset(
            source_manifest_paths=[teacher_train, validation],
            output_dir=tmp_path / "teacher_output",
        )


def test_hflip_normalized_coordinates_handle_half_ties_and_boundaries(tmp_path: Path) -> None:
    half_tie = _source_dataset(
        tmp_path / "half_tie",
        sequence_id="half_tie",
        split="train",
        color=51,
        width=5,
        x_normalized=0.1,
    )
    boundary = _source_dataset(
        tmp_path / "boundary",
        sequence_id="boundary",
        split="train",
        color=52,
        width=5,
        x_normalized=1.0,
    )
    validation = _source_dataset(
        tmp_path / "validation",
        sequence_id="validation",
        split="validation",
        color=53,
    )
    output = tmp_path / "student"
    build_student_dataset(
        source_manifest_paths=[half_tie, boundary, validation],
        output_dir=output,
        frame_transfer_mode="copy",
    )
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]

    half_view = next(
        sample
        for sample in samples
        if sample["source_sequence_id"] == "half_tie"
        and sample["augmentation"]["variant"] == "hflip"
    )
    half_landmark = half_view["hand"]["feature_landmarks"][0]
    assert half_landmark["u_px"] == 3
    assert half_landmark["x_normalized"] == pytest.approx(3 / 5)
    assert normalized_to_pixel(
        half_landmark["x_normalized"],
        half_landmark["y_normalized"],
        width=5,
        height=4,
    ) == (3, half_landmark["v_px"])

    boundary_view = next(
        sample
        for sample in samples
        if sample["source_sequence_id"] == "boundary"
        and sample["augmentation"]["variant"] == "hflip"
    )
    boundary_landmark = boundary_view["hand"]["feature_landmarks"][0]
    assert boundary_landmark["u_px"] == 0
    assert boundary_landmark["x_normalized"] == 0.0
    assert normalized_to_pixel(
        boundary_landmark["x_normalized"],
        boundary_landmark["y_normalized"],
        width=5,
        height=4,
    ) == (0, boundary_landmark["v_px"])
