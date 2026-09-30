from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from fingertip_depth.constants import HAND_LANDMARK_NAMES
from fingertip_depth.student_dataset import (
    STUDENT_DATASET_FORMAT,
    STUDENT_DATASET_FORMAT_VERSION,
    STUDENT_SAMPLE_SCHEMA_VERSION,
)
from fingertip_depth.student_model import FingertipDepthStudent, StudentModelConfig
from fingertip_depth.student_training import (
    FingertipStudentDataset,
    LoadedStudentCorpus,
    StudentTrainingConfig,
    StudentTrainingSample,
    TeacherSpikeFilterConfig,
    _autocast_enabled,
    _build_optimizer,
    _save_checkpoint,
    _train_one_epoch,
    apply_teacher_spike_filter,
    load_student_corpus,
    regression_metrics,
    train_student_transformer,
)
from fingertip_depth.video_cache import sha256_file

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_student_transformer.py"
_SPEC = importlib.util.spec_from_file_location("train_student_transformer_for_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
train_student_transformer_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_student_transformer_script)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    manifest_path = root / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, sha256_file(manifest_path)


def _fake_corpus(root: Path) -> dict[str, Any]:
    """Create a tiny final Phase 7 artifact without invoking either teacher."""

    root.mkdir(parents=True)
    frames = root / "frames"
    frames.mkdir()
    landmark_indices = (5, 6, 7, 8)
    rows: list[dict[str, Any]] = []
    specifications = (
        ("train:000000:aug=identity", "train", "train_sequence", "identity", 0, 0.20),
        ("train:000000:aug=hflip", "train", "train_sequence", "hflip", 0, 0.20),
        (
            "validation:000000:aug=identity",
            "validation",
            "validation_sequence",
            "identity",
            0,
            0.35,
        ),
    )
    for sample_number, (sample_id, split, sequence, variant, frame_index, depth) in enumerate(
        specifications
    ):
        source_sample_id = f"{sequence}:{frame_index:06d}:hand0"
        # A solid BGR image makes RGB channel-order assertions independent of interpolation.
        bgr = np.empty((3, 5, 3), dtype=np.uint8)
        bgr[...] = (10 + sample_number, 20 + sample_number, 30 + sample_number)
        image_path = frames / f"sample_{sample_number}.png"
        assert cv2.imwrite(str(image_path), bgr)
        landmarks = []
        for index in landmark_indices:
            landmarks.append(
                {
                    "landmark_index": index,
                    "landmark_name": HAND_LANDMARK_NAMES[index],
                    "x_normalized": index / 20.0,
                    "y_normalized": (20 - index) / 20.0,
                    # Deliberately non-numeric. Loading succeeds only if relative-z is
                    # completely ignored rather than parsed and subsequently discarded.
                    "z_mediapipe_relative": {"poison": f"landmark-{index}"},
                    "u_px": 1,
                    "v_px": 1,
                    "in_frame": True,
                }
            )
        rows.append(
            {
                "schema_version": STUDENT_SAMPLE_SCHEMA_VERSION,
                "sample_id": sample_id,
                "source_sample_id": source_sample_id,
                "split": split,
                "source_sequence_id": sequence,
                "frame_index": frame_index,
                "image": {
                    "relative_path": image_path.relative_to(root).as_posix(),
                    "width": 5,
                    "height": 3,
                    "png_sha256": sha256_file(image_path),
                },
                "hand": {"hand_index": 0, "feature_landmarks": landmarks},
                "target": {
                    "landmark_index": 8,
                    "landmark_name": "INDEX_FINGER_TIP",
                    "z_teacher_m": depth,
                    "depth_sampling": "single_pixel",
                    "depth_valid": True,
                },
                "augmentation": {
                    "variant": variant,
                    "source_sample_id": source_sample_id,
                },
            }
        )

    samples_path = root / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    targets_path = root / "targets.csv"
    with targets_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("sample_id", "z_teacher_m"))
        writer.writerows((row["sample_id"], row["target"]["z_teacher_m"]) for row in rows)
    splits = root / "splits"
    splits.mkdir()
    train_split_path = splits / "train.txt"
    validation_split_path = splits / "validation.txt"
    train_split_path.write_text(
        "".join(f"{row['sample_id']}\n" for row in rows if row["split"] == "train"),
        encoding="utf-8",
    )
    validation_split_path.write_text(
        "".join(f"{row['sample_id']}\n" for row in rows if row["split"] == "validation"),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "format": STUDENT_DATASET_FORMAT,
        "format_version": STUDENT_DATASET_FORMAT_VERSION,
        "sample_schema_version": STUDENT_SAMPLE_SCHEMA_VERSION,
        "label_type": "pseudo_label",
        "ground_truth": False,
        "feature_landmarks": {
            "indices": list(landmark_indices),
            "names": [HAND_LANDMARK_NAMES[index] for index in landmark_indices],
        },
        "target_landmark": {
            "index": 8,
            "name": "INDEX_FINGER_TIP",
            "depth_sampling": "single_pixel",
        },
        "split_policy": {
            "unit": "source video sequence",
            "random_frame_split": False,
            "train_source_sequences": ["train_sequence"],
            "validation_source_sequences": ["validation_sequence"],
            "validation_augmented": False,
        },
        "counts": {
            "samples_total": 3,
            "train_samples_total": 2,
            "validation_samples_total": 1,
        },
        "artifacts": {
            "samples_jsonl": {
                "relative_path": samples_path.relative_to(root).as_posix(),
                "sha256": sha256_file(samples_path),
            },
            "targets_csv": {
                "relative_path": targets_path.relative_to(root).as_posix(),
                "sha256": sha256_file(targets_path),
            },
            "train_split": {
                "relative_path": train_split_path.relative_to(root).as_posix(),
                "sha256": sha256_file(train_split_path),
            },
            "validation_split": {
                "relative_path": validation_split_path.relative_to(root).as_posix(),
                "sha256": sha256_file(validation_split_path),
            },
        },
    }
    manifest_path, manifest_sha256 = _write_manifest(root, manifest)
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "rows": rows,
        "samples_path": samples_path,
        "targets_path": targets_path,
    }


def _rewrite_as_chronological_corpus(
    artifact: dict[str, Any],
    *,
    validation_frame_index: int = 9,
) -> tuple[Path, str]:
    rows = [
        json.loads(line)
        for line in artifact["samples_path"].read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        is_validation = row["split"] == "validation"
        frame_index = validation_frame_index if is_validation else 0
        source_sample_id = f"shared:{frame_index:06d}:hand0"
        row["source_sequence_id"] = "shared"
        row["frame_index"] = frame_index
        row["source_sample_id"] = source_sample_id
        row["augmentation"]["source_sample_id"] = source_sample_id
    artifact["samples_path"].write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    artifact["manifest"]["split_policy"] = {
        "unit": "source video chronological frame tail",
        "random_frame_split": False,
        "temporal_order_preserved": True,
        "validation_tail_fraction": 0.1,
        "split_assignment_uses_teacher_depth": False,
        "train_source_sequences": ["shared"],
        "validation_source_sequences": ["shared"],
        "per_sequence": {
            "shared": {
                "source_frames_total": 10,
                "validation_frame_count": 1,
                "validation_start_frame_inclusive": 9,
                "train_frame_end_exclusive": 9,
                "accepted_train_identity_samples": 1,
                "accepted_validation_identity_samples": 1,
                "first_accepted_validation_frame": validation_frame_index,
                "last_accepted_validation_frame": validation_frame_index,
            }
        },
        "validation_augmented": False,
    }
    artifact["manifest"]["artifacts"]["samples_jsonl"]["sha256"] = sha256_file(
        artifact["samples_path"]
    )
    return _write_manifest(artifact["root"], artifact["manifest"])


def test_load_student_corpus_accepts_chronological_tail_split(tmp_path: Path) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    manifest_path, digest = _rewrite_as_chronological_corpus(artifact)

    corpus = load_student_corpus(
        manifest_path,
        expected_manifest_sha256=digest,
    )

    assert {sample.source_sequence_id for sample in corpus.train_samples} == {"shared"}
    assert {sample.source_sequence_id for sample in corpus.validation_samples} == {"shared"}
    assert {sample.frame_index for sample in corpus.train_samples} == {0}
    assert {sample.frame_index for sample in corpus.validation_samples} == {9}


def test_load_student_corpus_rejects_chronological_boundary_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    manifest_path, digest = _rewrite_as_chronological_corpus(
        artifact,
        validation_frame_index=8,
    )

    with pytest.raises(ValueError, match="chronological frame boundary"):
        load_student_corpus(
            manifest_path,
            expected_manifest_sha256=digest,
        )


def test_load_student_corpus_rejects_frame_beyond_source_video(
    tmp_path: Path,
) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    manifest_path, digest = _rewrite_as_chronological_corpus(
        artifact,
        validation_frame_index=10,
    )

    with pytest.raises(ValueError, match="source video frame count"):
        load_student_corpus(manifest_path, expected_manifest_sha256=digest)


def test_load_student_corpus_rejects_chronological_identity_count_mismatch(
    tmp_path: Path,
) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    _rewrite_as_chronological_corpus(artifact)
    boundary = artifact["manifest"]["split_policy"]["per_sequence"]["shared"]
    boundary["accepted_validation_identity_samples"] = 2
    manifest_path, digest = _write_manifest(artifact["root"], artifact["manifest"])

    with pytest.raises(ValueError, match="accepted-sample metadata"):
        load_student_corpus(manifest_path, expected_manifest_sha256=digest)


def _temporal_sample(
    *,
    sequence: str,
    split: str,
    frame_index: int,
    depth_m: float,
    variant: str = "identity",
    hand_index: int = 0,
    source_sample_id: str | None = None,
) -> StudentTrainingSample:
    source_id = source_sample_id or f"{sequence}:{frame_index:06d}:hand{hand_index}"
    return StudentTrainingSample(
        sample_id=f"{source_id}:aug={variant}",
        source_sample_id=source_id,
        split=split,  # type: ignore[arg-type]
        image_path=Path("/unused/filter-only.png"),
        image_png_sha256="0" * 64,
        width=5,
        height=3,
        landmark_xy=((0.5, 0.5),),
        target_depth_m=depth_m,
        source_sequence_id=sequence,
        hand_index=hand_index,
        frame_index=frame_index,
        augmentation_variant=variant,  # type: ignore[arg-type]
    )


def _temporal_corpus(
    train: list[StudentTrainingSample],
    validation: list[StudentTrainingSample],
) -> LoadedStudentCorpus:
    return LoadedStudentCorpus(
        manifest_path=Path("/unused/dataset_manifest.json"),
        manifest_sha256="a" * 64,
        manifest={},
        available_landmark_indices=(8,),
        selected_landmark_indices=(8,),
        train_samples=tuple(train),
        validation_samples=tuple(validation),
        artifact_sha256={},
    )


def _paired_train_stream(
    sequence: str,
    depths: list[float],
    *,
    start_frame: int = 0,
) -> list[StudentTrainingSample]:
    samples: list[StudentTrainingSample] = []
    for offset, depth in enumerate(depths):
        frame_index = start_frame + offset
        identity = _temporal_sample(
            sequence=sequence,
            split="train",
            frame_index=frame_index,
            depth_m=depth,
        )
        samples.extend(
            (
                identity,
                _temporal_sample(
                    sequence=sequence,
                    split="train",
                    frame_index=frame_index,
                    depth_m=depth,
                    variant="hflip",
                    source_sample_id=identity.source_sample_id,
                ),
            )
        )
    return samples


def test_teacher_spike_filter_rejects_isolated_and_two_frame_bursts_with_hflip() -> None:
    train = [
        *_paired_train_stream(
            "single_spike",
            [0.30, 0.31, 0.32, 0.90, 0.33, 0.34, 0.35, 0.36],
        ),
        *_paired_train_stream(
            "two_frame_burst",
            [0.30, 0.31, 0.32, 1.20, 1.40, 0.33, 0.34, 0.35, 0.36],
        ),
    ]
    validation = [
        _temporal_sample(
            sequence="boundary_spike",
            split="validation",
            frame_index=frame,
            depth_m=depth,
        )
        for frame, depth in enumerate([0.95, 0.22, 0.23, 0.24, 0.25, 0.26])
    ]
    result = apply_teacher_spike_filter(
        _temporal_corpus(train, validation),
        TeacherSpikeFilterConfig(enabled=True),
    )

    rejected = {
        (row["source_sequence_id"], row["frame_index"])
        for row in result.report["rejected_observations"]
    }
    assert rejected == {
        ("single_spike", 3),
        ("two_frame_burst", 3),
        ("two_frame_burst", 4),
        ("boundary_spike", 0),
    }
    assert len(train) - len(result.train_samples) == 6
    assert len(validation) - len(result.validation_samples) == 1
    assert result.report["counts"]["identity_rejected"] == 4
    assert result.report["counts"]["views_removed_total"] == 7
    assert all(
        sample.frame_index not in {3, 4}
        for sample in result.train_samples
        if sample.source_sequence_id == "two_frame_burst"
    )


def test_teacher_depth_limit_removes_sustained_high_values_and_hflip() -> None:
    train = _paired_train_stream("high_plateau", [0.30, 0.80, 0.81, 0.82, 0.30])
    validation = [
        _temporal_sample(
            sequence="validation",
            split="validation",
            frame_index=frame,
            depth_m=depth,
        )
        for frame, depth in enumerate([0.30, 0.79, 0.80, 0.81, 0.30])
    ]
    result = apply_teacher_spike_filter(
        _temporal_corpus(train, validation),
        TeacherSpikeFilterConfig(enabled=False, max_depth_m=0.80),
    )

    assert {sample.frame_index for sample in result.train_samples} == {0, 4}
    assert {sample.frame_index for sample in result.validation_samples} == {0, 1, 4}
    assert result.report["counts"]["depth_limit_rejected"] == 5
    assert result.report["counts"]["temporal_spike_rejected"] == 0
    assert result.report["counts"]["views_removed_total"] == 8
    assert all(
        decision["reason"] == "max_depth_exceeded"
        for decision in result.report["rejected_observations"]
    )


def test_teacher_spike_filter_keeps_smooth_motion_and_does_not_cross_gaps() -> None:
    train = _paired_train_stream(
        "smooth",
        [0.30, 0.33, 0.36, 0.39, 0.36, 0.33, 0.30],
    )
    train.extend(
        _paired_train_stream(
            "short_after_gap",
            [1.20, 0.30, 0.31],
            start_frame=10,
        )
    )
    validation = [
        _temporal_sample(
            sequence="validation",
            split="validation",
            frame_index=frame,
            depth_m=depth,
        )
        for frame, depth in enumerate([0.20, 0.22, 0.24, 0.26])
    ]
    corpus = _temporal_corpus(train, validation)

    enabled = apply_teacher_spike_filter(corpus, TeacherSpikeFilterConfig(enabled=True))
    disabled = apply_teacher_spike_filter(corpus, TeacherSpikeFilterConfig(enabled=False))

    assert enabled.report["counts"]["identity_rejected"] == 0
    assert enabled.train_samples == corpus.train_samples
    assert disabled.train_samples == corpus.train_samples
    assert disabled.validation_samples == corpus.validation_samples
    assert disabled.included_train_ids_sha256 == enabled.included_train_ids_sha256


def test_teacher_spike_filter_rejects_mismatched_hflip_provenance() -> None:
    identity = _temporal_sample(
        sequence="train",
        split="train",
        frame_index=0,
        depth_m=0.30,
    )
    mismatched = _temporal_sample(
        sequence="train",
        split="train",
        frame_index=0,
        depth_m=0.31,
        variant="hflip",
        source_sample_id=identity.source_sample_id,
    )
    validation = [
        _temporal_sample(
            sequence="validation",
            split="validation",
            frame_index=frame,
            depth_m=0.30,
        )
        for frame in range(4)
    ]

    with pytest.raises(ValueError, match="augmented view differs"):
        apply_teacher_spike_filter(
            _temporal_corpus([identity, mismatched], validation),
            TeacherSpikeFilterConfig(enabled=True),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"frame_radius": 0}, "frame_radius"),
        ({"frame_radius": 2, "min_neighbors": 5}, "min_neighbors"),
        ({"absolute_floor_m": 0.0}, "thresholds"),
        ({"max_depth_m": 0.0}, "max_depth_m"),
    ),
)
def test_teacher_spike_filter_config_rejects_invalid_values(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TeacherSpikeFilterConfig(**kwargs)


def test_load_student_corpus_validates_split_and_selects_xy_without_reading_z(
    tmp_path: Path,
) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")

    corpus = load_student_corpus(
        artifact["manifest_path"],
        expected_manifest_sha256=artifact["manifest_sha256"],
        landmark_indices=(8, 5),
    )

    assert corpus.manifest_sha256 == artifact["manifest_sha256"]
    assert corpus.available_landmark_indices == (5, 6, 7, 8)
    assert corpus.selected_landmark_indices == (8, 5)
    assert [sample.sample_id for sample in corpus.train_samples] == [
        "train:000000:aug=identity",
        "train:000000:aug=hflip",
    ]
    assert [sample.sample_id for sample in corpus.validation_samples] == [
        "validation:000000:aug=identity"
    ]
    assert corpus.train_samples[0].landmark_xy[0] == pytest.approx((0.4, 0.6))
    assert corpus.train_samples[0].landmark_xy[1] == pytest.approx((0.25, 0.75))
    assert not hasattr(corpus.train_samples[0], "z_mediapipe_relative")


def test_load_student_corpus_rejects_manifest_and_artifact_hash_mismatches(
    tmp_path: Path,
) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_student_corpus(
            artifact["manifest_path"],
            expected_manifest_sha256="0" * 64,
        )

    artifact["targets_path"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch: targets_csv"):
        load_student_corpus(
            artifact["manifest_path"],
            expected_manifest_sha256=artifact["manifest_sha256"],
        )


def test_load_student_corpus_rejects_path_escape_and_sequence_split_overlap(
    tmp_path: Path,
) -> None:
    artifact = _fake_corpus(tmp_path / "path_dataset")
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    artifact["manifest"]["artifacts"]["samples_jsonl"] = {
        "relative_path": "../outside.jsonl",
        "sha256": sha256_file(outside),
    }
    manifest_path, digest = _write_manifest(artifact["root"], artifact["manifest"])
    with pytest.raises(ValueError, match="escapes its root"):
        load_student_corpus(manifest_path, expected_manifest_sha256=digest)

    overlap = _fake_corpus(tmp_path / "split_dataset")
    overlap["manifest"]["split_policy"]["validation_source_sequences"] = ["train_sequence"]
    manifest_path, digest = _write_manifest(overlap["root"], overlap["manifest"])
    with pytest.raises(ValueError, match="disjoint"):
        load_student_corpus(manifest_path, expected_manifest_sha256=digest)


def test_load_student_corpus_rejects_landmarks_absent_from_manifest(tmp_path: Path) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")

    with pytest.raises(ValueError, match="requested landmarks are not stored"):
        load_student_corpus(
            artifact["manifest_path"],
            expected_manifest_sha256=artifact["manifest_sha256"],
            landmark_indices=(4, 8),
        )


def test_dataset_returns_rgb_and_centered_xy_only(tmp_path: Path) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    corpus = load_student_corpus(
        artifact["manifest_path"],
        expected_manifest_sha256=artifact["manifest_sha256"],
        landmark_indices=(8, 5),
    )
    dataset = FingertipStudentDataset(
        corpus.train_samples[:1],
        image_size=4,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        preload_images=True,
        verify_png_sha256=True,
    )

    item = dataset[0]

    assert item["pixel_values"].shape == (3, 4, 4)
    torch.testing.assert_close(
        item["pixel_values"][:, 0, 0],
        torch.tensor((30.0, 20.0, 10.0)) / 255.0,
    )
    assert item["landmark_coordinates"].shape == (2, 2)
    torch.testing.assert_close(
        item["landmark_coordinates"],
        torch.tensor(((-0.2, 0.2), (-0.5, 0.5))),
    )
    assert "z_mediapipe_relative" not in item
    assert set(item) == {
        "pixel_values",
        "landmark_coordinates",
        "target_depth_m",
        "sample_id",
        "source_sequence_id",
        "frame_index",
        "augmentation_variant",
    }


def test_dataset_verifies_png_hash_when_loading_lazily(tmp_path: Path) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    corpus = load_student_corpus(
        artifact["manifest_path"],
        expected_manifest_sha256=artifact["manifest_sha256"],
    )
    corpus.train_samples[0].image_path.write_bytes(b"tampered")
    dataset = FingertipStudentDataset(
        corpus.train_samples,
        image_size=4,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        preload_images=False,
        verify_png_sha256=True,
    )

    with pytest.raises(ValueError, match="sample PNG SHA-256 mismatch"):
        dataset[0]


def test_regression_metrics_are_exact_for_known_values() -> None:
    prediction = np.asarray((1.0, 2.0, 3.0))
    target = np.asarray((1.5, 1.5, 2.0))

    metrics = regression_metrics(prediction, target)

    assert metrics["count"] == 3
    assert metrics["mse_m2"] == pytest.approx(0.5)
    assert metrics["rmse_m"] == pytest.approx(math.sqrt(0.5))
    assert metrics["rmse_cm"] == pytest.approx(100.0 * math.sqrt(0.5))
    assert metrics["mae_m"] == pytest.approx(2.0 / 3.0)
    assert metrics["median_absolute_error_m"] == pytest.approx(0.5)
    assert metrics["p95_absolute_error_m"] == pytest.approx(0.95)
    assert metrics["bias_m"] == pytest.approx(1.0 / 3.0)
    assert metrics["max_absolute_error_m"] == pytest.approx(1.0)
    assert metrics["negative_prediction_count"] == 0


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (np.asarray([]), np.asarray([])),
        (np.asarray([[1.0]]), np.asarray([[1.0]])),
        (np.asarray([1.0]), np.asarray([1.0, 2.0])),
        (np.asarray([np.nan]), np.asarray([1.0])),
    ],
)
def test_regression_metrics_reject_invalid_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        regression_metrics(prediction, target)


class _TinyImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 6) -> None:
        super().__init__()
        self.num_features = embedding_dim
        self.projection = nn.Conv2d(3, embedding_dim, kernel_size=2, stride=2)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).flatten(2).transpose(1, 2)


def _tiny_student() -> FingertipDepthStudent:
    return FingertipDepthStudent(
        StudentModelConfig(
            image_encoder_name="fake-tiny",
            pretrained_image_encoder=False,
            landmark_indices=(5, 8),
            fusion_layers=1,
            fusion_heads=3,
            fusion_mlp_ratio=2.0,
            dropout=0.0,
        ),
        initial_depth_bias_m=0.3,
        image_encoder=_TinyImageEncoder(),
    )


def test_optimizer_groups_lrs_and_excludes_all_frozen_type_tokens() -> None:
    model = _tiny_student()
    config = StudentTrainingConfig(
        encoder_learning_rate=2e-5,
        head_learning_rate=3e-4,
        weight_decay=0.2,
        precision="float32",
        num_workers=0,
    )
    optimizer = _build_optimizer(model, config)
    optimized_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimized_ids = [id(parameter) for parameter in optimized_parameters]

    assert len(optimized_ids) == len(set(optimized_ids))
    assert set(optimized_ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert all(
        id(parameter) not in optimized_ids
        for parameter in model.landmark_type_tokens.frozen_token_parameters()
    )
    assert all(
        id(parameter) in optimized_ids
        for parameter in model.landmark_type_tokens.trainable_token_parameters()
    )
    assert {group["lr"] for group in optimizer.param_groups} == {2e-5, 3e-4}
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.2}

    frozen_before = [
        parameter.detach().clone()
        for parameter in model.landmark_type_tokens.frozen_token_parameters()
    ]
    final = model.depth_head[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.weight.fill_(0.1)
    loss = model(torch.randn(2, 3, 4, 4), torch.randn(2, 2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(
            frozen_before,
            model.landmark_type_tokens.frozen_token_parameters(),
            strict=True,
        )
    )


class _ConstantDepthModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        del landmarks
        return self.bias.expand(images.shape[0])


def test_train_epoch_uses_sample_weighted_fp32_mse() -> None:
    rows = [
        {
            "pixel_values": torch.zeros(3, 2, 2),
            "landmark_coordinates": torch.zeros(1, 2),
            "target_depth_m": torch.tensor(target),
        }
        for target in (0.0, 2.0, 4.0)
    ]
    loader = DataLoader(rows, batch_size=2, shuffle=False)
    model = _ConstantDepthModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = StudentTrainingConfig(
        precision="float32",
        num_workers=0,
        freeze_image_encoder=False,
    )

    metrics = _train_one_epoch(
        model,  # type: ignore[arg-type]
        loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=torch.device("cpu"),
    )

    assert metrics["mse_m2"] == pytest.approx(11.0 / 3.0)


def test_autocast_is_bfloat16_cuda_only() -> None:
    bfloat16 = StudentTrainingConfig(precision="bfloat16")
    float32 = StudentTrainingConfig(precision="float32")

    assert not _autocast_enabled(bfloat16, torch.device("cpu"))
    assert _autocast_enabled(bfloat16, torch.device("cuda:0"))
    assert not _autocast_enabled(float32, torch.device("cuda:0"))


def test_checkpoint_round_trip_contains_model_and_provenance(tmp_path: Path) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    corpus = load_student_corpus(
        artifact["manifest_path"],
        expected_manifest_sha256=artifact["manifest_sha256"],
        landmark_indices=(5, 8),
    )
    model = _tiny_student()
    model_config = model.config
    training_config = StudentTrainingConfig(precision="float32", num_workers=0)
    checkpoint_path = tmp_path / "checkpoint.pt"

    _save_checkpoint(
        checkpoint_path,
        model=model,
        epoch=3,
        validation_metrics={"mse_m2": 0.0123},
        corpus=corpus,
        model_config=model_config,
        training_config=training_config,
        spike_filter_config=TeacherSpikeFilterConfig(enabled=True),
        selection_provenance={"included_train_ids_sha256": "b" * 64},
        initial_encoder_state_sha256="a" * 64,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert payload["format"] == "fingertip-depth-student-checkpoint"
    assert payload["epoch"] == 3
    assert payload["dataset_manifest_sha256"] == artifact["manifest_sha256"]
    assert payload["model_config"]["landmark_indices"] == [5, 8]
    assert payload["training_config"]["precision"] == "float32"
    assert payload["teacher_spike_filter_config"]["enabled"] is True
    assert payload["data_selection"]["included_train_ids_sha256"] == "b" * 64
    assert set(payload["model_state_dict"]) == set(model.state_dict())
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()


def test_training_writes_filtered_and_raw_audit_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fake_corpus(tmp_path / "dataset")
    model_config = StudentModelConfig(
        image_encoder_name="fake-tiny",
        pretrained_image_encoder=False,
        landmark_indices=(5, 8),
        fusion_layers=1,
        fusion_heads=3,
        fusion_mlp_ratio=2.0,
        dropout=0.0,
    )

    def fake_student(
        config: StudentModelConfig,
        *,
        initial_depth_bias_m: float,
    ) -> FingertipDepthStudent:
        model = FingertipDepthStudent(
            config,
            initial_depth_bias_m=initial_depth_bias_m,
            image_encoder=_TinyImageEncoder(),
        )
        # Production deliberately zero-initializes the last weight. The tiny
        # two-sample run needs immediate upstream gradients so its best epoch
        # can exercise the learned-token postcondition deterministically.
        final = model.depth_head[-1]
        assert isinstance(final, nn.Linear)
        with torch.no_grad():
            final.weight.fill_(0.01)
        return model

    monkeypatch.setattr(
        "fingertip_depth.student_training.FingertipDepthStudent",
        fake_student,
    )
    output_dir = tmp_path / "run"
    result = train_student_transformer(
        dataset_manifest_path=artifact["manifest_path"],
        expected_dataset_manifest_sha256=artifact["manifest_sha256"],
        output_dir=output_dir,
        model_config=model_config,
        training_config=StudentTrainingConfig(
            epochs=2,
            batch_size=1,
            encoder_learning_rate=1e-3,
            head_learning_rate=1e-3,
            warmup_fraction=0.0,
            early_stopping_patience=0,
            image_size=4,
            num_workers=0,
            precision="float32",
            preload_images=False,
        ),
        spike_filter_config=TeacherSpikeFilterConfig(enabled=True),
        device_name="cpu",
    )

    assert result["format_version"] == 2
    assert result["dataset"]["raw_train_samples"] == 2
    assert result["dataset"]["selected_train_samples"] == 2
    assert result["dataset"]["raw_validation_samples"] == 1
    assert result["dataset"]["selected_validation_samples"] == 1
    assert result["dataset"]["teacher_spike_filter"]["counts"]["identity_rejected"] == 0
    assert result["results"]["checkpoint_selection"]["cohort"] == "filtered_validation"
    assert result["results"]["best_validation_filtered"]["count"] == 1
    assert result["results"]["raw_validation_posthoc_diagnostic"]["count"] == 1
    expected_artifacts = {
        "best_checkpoint",
        "last_checkpoint",
        "history",
        "validation_predictions_filtered",
        "validation_predictions_raw",
        "teacher_spike_filter",
        "identity_decisions",
        "included_train",
        "included_validation",
    }
    assert set(result["artifacts"]) == expected_artifacts
    for metadata in result["artifacts"].values():
        path = output_dir / metadata["relative_path"]
        assert path.is_file()
        assert sha256_file(path) == metadata["sha256"]
    checkpoint = torch.load(
        output_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["format_version"] == 2
    assert checkpoint["teacher_spike_filter_config"]["enabled"] is True
    assert checkpoint["data_selection"]["checkpoint_selection_scope"] == ("filtered_validation")


def test_cli_main_parses_and_forwards_training_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_train_student_transformer(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "results": {
                "best_epoch": 2,
                "epochs_completed": 3,
                "best_validation": {"mse_m2": 0.01},
                "validation_baselines": {},
            }
        }

    monkeypatch.setattr(
        train_student_transformer_script,
        "train_student_transformer",
        fake_train_student_transformer,
    )
    output_dir = tmp_path / "output"
    result = train_student_transformer_script.main(
        [
            "--dataset-manifest",
            str(tmp_path / "dataset.json"),
            "--expected-dataset-manifest-sha256",
            "b" * 64,
            "--output-dir",
            str(output_dir),
            "--landmarks",
            "INDEX_FINGER_TIP,5",
            "--image-encoder",
            "fake-encoder",
            "--no-pretrained",
            "--fusion-layers",
            "1",
            "--fusion-heads",
            "3",
            "--epochs",
            "4",
            "--batch-size",
            "2",
            "--precision",
            "float32",
            "--device",
            "cpu",
            "--freeze-image-encoder",
            "--no-preload-images",
            "--skip-image-png-sha256",
            "--exclude-teacher-spikes",
            "--teacher-spike-frame-radius",
            "4",
            "--teacher-spike-max-frame-gap",
            "2",
            "--teacher-spike-min-neighbors",
            "4",
            "--teacher-spike-absolute-floor-m",
            "0.12",
            "--teacher-spike-relative-floor-fraction",
            "0.4",
            "--teacher-spike-mad-multiplier",
            "7",
            "--teacher-spike-mad-scale",
            "1.5",
            "--teacher-max-depth-m",
            "0.8",
        ]
    )

    assert result == 0
    assert captured["dataset_manifest_path"] == tmp_path / "dataset.json"
    assert captured["expected_dataset_manifest_sha256"] == "b" * 64
    assert captured["output_dir"] == output_dir
    assert captured["device_name"] == "cpu"
    assert captured["model_config"].landmark_indices == (8, 5)
    assert captured["model_config"].image_encoder_name == "fake-encoder"
    assert captured["model_config"].pretrained_image_encoder is False
    assert captured["training_config"].epochs == 4
    assert captured["training_config"].batch_size == 2
    assert captured["training_config"].precision == "float32"
    assert captured["training_config"].freeze_image_encoder is True
    assert captured["training_config"].preload_images is False
    assert captured["training_config"].verify_image_png_sha256 is False
    assert captured["spike_filter_config"] == TeacherSpikeFilterConfig(
        enabled=True,
        frame_radius=4,
        max_frame_gap=2,
        min_neighbors=4,
        absolute_floor_m=0.12,
        relative_floor_fraction=0.4,
        mad_multiplier=7.0,
        mad_scale=1.5,
        max_depth_m=0.8,
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["best_epoch"] == 2
