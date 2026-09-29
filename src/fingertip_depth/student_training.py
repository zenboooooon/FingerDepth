'疑似ラベルデータの検証、任意の教師外れ値除外、生徒モデルの訓練と評価、チェックポイントや監査記録の保存を担当します。'

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import write_json
from .constants import DEFAULT_FEATURE_LANDMARK_INDICES, HAND_LANDMARK_NAMES
from .student_dataset import (
    STUDENT_DATASET_FORMAT,
    STUDENT_DATASET_FORMAT_VERSION,
    STUDENT_SAMPLE_SCHEMA_VERSION,
    _ceil_fractional_count,
)
from .student_model import FingertipDepthStudent, StudentModelConfig
from .video_cache import sha256_file

TRAINING_RUN_FORMAT = "fingertip-depth-student-training"
TRAINING_RUN_FORMAT_VERSION = 2
STUDENT_CHECKPOINT_FORMAT_VERSION = 2
TEACHER_SPIKE_FILTER_FORMAT = "fingertip-depth-teacher-spike-filter"
TEACHER_SPIKE_FILTER_FORMAT_VERSION = 1
DEFAULT_DATASET_MANIFEST_SHA256 = "45d3ba7b6865d0bae91e79ef977fc5e8dba7f9f3a1c63c690dd2c7476093bbdc"
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)


# 生徒モデルの学習回数、最適化、分割、再現性に関する設定です。
@dataclass(frozen=True, slots=True)
class StudentTrainingConfig:
    """Optimization and deterministic preprocessing settings."""

    epochs: int = 20
    batch_size: int = 32
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 6
    image_size: int = 224
    image_mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN
    image_std: tuple[float, float, float] = DEFAULT_IMAGE_STD
    num_workers: int = 4
    seed: int = 20260925
    precision: Literal["float32", "bfloat16"] = "bfloat16"
    freeze_image_encoder: bool = False
    preload_images: bool = True
    verify_image_png_sha256: bool = True

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.encoder_learning_rate <= 0 or self.head_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image mean/std must contain three channels")
        if any(not math.isfinite(value) for value in (*self.image_mean, *self.image_std)):
            raise ValueError("image mean/std values must be finite")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("image std values must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")

    # 主要なフィールドを、JSONへ保存できる辞書に変換します。
    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_mean"] = list(self.image_mean)
        value["image_std"] = list(self.image_std)
        return value


# 教師深度の時系列スパイクを除外する条件を保持します。
@dataclass(frozen=True, slots=True)
class TeacherSpikeFilterConfig:
    """Deterministic run-level filtering of isolated teacher-depth excursions."""

    enabled: bool = False
    frame_radius: int = 3
    max_frame_gap: int = 1
    min_neighbors: int = 3
    absolute_floor_m: float = 0.15
    relative_floor_fraction: float = 0.50
    mad_multiplier: float = 6.0
    mad_scale: float = 1.4826

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if self.frame_radius <= 0:
            raise ValueError("teacher spike frame_radius must be positive")
        if self.max_frame_gap <= 0:
            raise ValueError("teacher spike max_frame_gap must be positive")
        if self.min_neighbors <= 0 or self.min_neighbors > 2 * self.frame_radius:
            raise ValueError("teacher spike min_neighbors must be in [1, 2 * frame_radius]")
        values = (
            self.absolute_floor_m,
            self.relative_floor_fraction,
            self.mad_multiplier,
            self.mad_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("teacher spike thresholds must be finite and positive")

    # 主要なフィールドを、JSONへ保存できる辞書に変換します。
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# 画像、ランドマーク、教師深度、出典からなる学習標本です。
@dataclass(frozen=True, slots=True)
class StudentTrainingSample:
    sample_id: str
    source_sample_id: str
    split: Literal["train", "validation"]
    image_path: Path
    image_png_sha256: str
    width: int
    height: int
    landmark_xy: tuple[tuple[float, float], ...]
    target_depth_m: float
    source_sequence_id: str
    hand_index: int
    frame_index: int
    augmentation_variant: Literal["identity", "hflip"]


# 読み込みと検証を終えた学習コーパスおよび分割情報を保持します。
@dataclass(frozen=True, slots=True)
class LoadedStudentCorpus:
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    available_landmark_indices: tuple[int, ...]
    selected_landmark_indices: tuple[int, ...]
    train_samples: tuple[StudentTrainingSample, ...]
    validation_samples: tuple[StudentTrainingSample, ...]
    artifact_sha256: Mapping[str, str]


# スパイク判定後に残った標本と、除外に関する記録を保持します。
@dataclass(frozen=True, slots=True)
class TeacherSpikeFilterResult:
    """Exact retained cohorts and identity-only temporal filter decisions."""

    config: TeacherSpikeFilterConfig
    train_samples: tuple[StudentTrainingSample, ...]
    validation_samples: tuple[StudentTrainingSample, ...]
    decisions: tuple[Mapping[str, Any], ...]
    report: Mapping[str, Any]
    included_train_ids_sha256: str
    included_validation_ids_sha256: str


# JSONファイルを読み込みます。
def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


# JSON Linesを読み込み、行ごとのレコードを返します。
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


# 値が正しい形式のSHA-256であることを検証します。
def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    return digest


# 指定ファイルが成果物ルートの外へ逸脱しないことを検証します。
def _contained_file(root: Path, relative_path: object) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise ValueError(f"dataset path must be relative: {relative}")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"dataset path escapes its root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


# 選択ランドマークが学習コーパスとモデル設定で一致するか検証します。
def _validate_landmark_selection(indices: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(int(index) for index in indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected landmark indices must be non-empty and unique")
    if any(not 0 <= index < len(HAND_LANDMARK_NAMES) for index in selected):
        raise ValueError("selected landmark index is outside MediaPipe's 0..20 range")
    return selected


# データセットのマニフェストと標本を読み込み、ハッシュや参照ファイルを検証します。
def load_student_corpus(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    landmark_indices: Sequence[int] = DEFAULT_FEATURE_LANDMARK_INDICES,
) -> LoadedStudentCorpus:
    """Load and validate the immutable multi-sequence Phase 8 input corpus."""

    manifest_path = manifest_path.resolve()
    selected = _validate_landmark_selection(landmark_indices)
    expected = _require_sha256(expected_manifest_sha256, field="expected dataset manifest SHA-256")
    observed = sha256_file(manifest_path)
    if observed != expected:
        raise ValueError(
            f"dataset manifest SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("format") != STUDENT_DATASET_FORMAT:
        raise ValueError("unsupported student dataset format")
    if manifest.get("format_version") != STUDENT_DATASET_FORMAT_VERSION:
        raise ValueError("unsupported student dataset format version")
    if manifest.get("sample_schema_version") != STUDENT_SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported student sample schema version")
    if manifest.get("ground_truth") is not False or manifest.get("label_type") != "pseudo_label":
        raise ValueError("Phase 8 input must remain explicitly marked as pseudo-labels")

    feature_meta = manifest.get("feature_landmarks")
    if not isinstance(feature_meta, Mapping):
        raise TypeError("dataset feature_landmarks must be an object")
    available = tuple(int(index) for index in feature_meta.get("indices", []))
    available_names = tuple(str(name) for name in feature_meta.get("names", []))
    if len(available) != len(available_names) or not available:
        raise ValueError("dataset feature landmark metadata is inconsistent")
    for index, name in zip(available, available_names, strict=True):
        if not 0 <= index < len(HAND_LANDMARK_NAMES) or HAND_LANDMARK_NAMES[index] != name:
            raise ValueError("dataset feature landmark index/name mapping is invalid")
    missing = [index for index in selected if index not in available]
    if missing:
        raise ValueError(
            f"requested landmarks are not stored in this dataset: {missing}; "
            f"available indices are {list(available)}"
        )
    target_meta = manifest.get("target_landmark")
    if not isinstance(target_meta, Mapping):
        raise TypeError("dataset target_landmark must be an object")
    if (
        int(target_meta.get("index", -1)) != 8
        or target_meta.get("name") != HAND_LANDMARK_NAMES[8]
        or target_meta.get("depth_sampling") != "single_pixel"
    ):
        raise ValueError("Phase 8 requires the single-pixel INDEX_FINGER_TIP target")

    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, Mapping):
        raise TypeError("dataset split_policy must be an object")
    split_unit = str(split_policy.get("unit", ""))
    if split_policy.get("random_frame_split") is not False:
        raise ValueError("random frame splitting is not allowed")
    if split_policy.get("validation_augmented") is not False:
        raise ValueError("validation must remain unaugmented")
    train_sequences = {str(value) for value in split_policy.get("train_source_sequences", [])}
    validation_sequences = {
        str(value) for value in split_policy.get("validation_source_sequences", [])
    }
    chronological_boundaries: dict[str, int] = {}
    chronological_frame_totals: dict[str, int] = {}
    chronological_metadata: dict[str, Mapping[str, Any]] = {}
    if split_unit == "source video sequence":
        if (
            not train_sequences
            or not validation_sequences
            or train_sequences & validation_sequences
        ):
            raise ValueError(
                "sequence-split train and validation source sets must be non-empty and disjoint"
            )
    elif split_unit == "source video chronological frame tail":
        if (
            not train_sequences
            or train_sequences != validation_sequences
            or split_policy.get("temporal_order_preserved") is not True
            or split_policy.get("split_assignment_uses_teacher_depth") is not False
        ):
            raise ValueError("chronological tail split policy is inconsistent")
        fraction = float(split_policy.get("validation_tail_fraction", math.nan))
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("chronological validation tail fraction must be in (0, 1)")
        per_sequence = split_policy.get("per_sequence")
        if not isinstance(per_sequence, Mapping) or set(map(str, per_sequence)) != train_sequences:
            raise ValueError("chronological split boundaries differ from source sequences")
        for sequence_id in sorted(train_sequences):
            boundary = per_sequence.get(sequence_id)
            if not isinstance(boundary, Mapping):
                raise TypeError("chronological per-sequence boundary must be an object")
            source_frames_total = int(boundary.get("source_frames_total", -1))
            validation_frame_count = int(boundary.get("validation_frame_count", -1))
            validation_start = int(boundary.get("validation_start_frame_inclusive", -1))
            expected_count = _ceil_fractional_count(source_frames_total, fraction)
            if (
                source_frames_total <= 0
                or validation_frame_count != expected_count
                or validation_start != source_frames_total - expected_count
                or int(boundary.get("train_frame_end_exclusive", -1)) != validation_start
                or int(boundary.get("accepted_train_identity_samples", 0)) <= 0
                or int(boundary.get("accepted_validation_identity_samples", 0)) <= 0
            ):
                raise ValueError(f"chronological split boundary is inconsistent: {sequence_id}")
            chronological_boundaries[sequence_id] = validation_start
            chronological_frame_totals[sequence_id] = source_frames_total
            chronological_metadata[sequence_id] = boundary
    else:
        raise ValueError(f"unsupported Phase 8 split unit: {split_unit!r}")

    root = manifest_path.parent
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("dataset artifacts must be an object")
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for key in ("samples_jsonl", "targets_csv", "train_split", "validation_split"):
        entry = artifacts.get(key)
        if not isinstance(entry, Mapping):
            raise TypeError(f"dataset artifact {key} must be an object")
        path = _contained_file(root, entry.get("relative_path"))
        digest = _require_sha256(entry.get("sha256"), field=f"artifact {key} SHA-256")
        if sha256_file(path) != digest:
            raise ValueError(f"dataset artifact SHA-256 mismatch: {key}")
        artifact_paths[key] = path
        artifact_hashes[key] = digest

    rows = _read_jsonl(artifact_paths["samples_jsonl"])
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or int(counts.get("samples_total", -1)) != len(rows):
        raise ValueError("dataset sample count differs from samples.jsonl")
    sample_ids: set[str] = set()
    samples: list[StudentTrainingSample] = []
    for row in rows:
        if row.get("schema_version") != STUDENT_SAMPLE_SCHEMA_VERSION:
            raise ValueError("sample schema version differs from the manifest")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in sample_ids:
            raise ValueError("sample IDs must be non-empty and globally unique")
        sample_ids.add(sample_id)
        source_sample_id = str(row.get("source_sample_id", ""))
        if not source_sample_id:
            raise ValueError("sample source_sample_id must be non-empty")
        split = str(row.get("split", ""))
        if split not in {"train", "validation"}:
            raise ValueError(f"unsupported sample split: {split}")
        source_sequence_id = str(row.get("source_sequence_id", ""))
        frame_index = int(row.get("frame_index", -1))
        if frame_index < 0:
            raise ValueError("sample frame_index must be non-negative")
        expected_sequences = train_sequences if split == "train" else validation_sequences
        if source_sequence_id not in expected_sequences:
            raise ValueError("sample source sequence differs from the split policy")
        if split_unit == "source video chronological frame tail":
            if frame_index >= chronological_frame_totals[source_sequence_id]:
                raise ValueError("sample frame_index exceeds its source video frame count")
            expected_split = (
                "validation"
                if frame_index >= chronological_boundaries[source_sequence_id]
                else "train"
            )
            if split != expected_split:
                raise ValueError("sample split differs from its chronological frame boundary")

        image = row.get("image")
        if not isinstance(image, Mapping):
            raise TypeError("sample image must be an object")
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("sample image dimensions must be positive")
        image_path = _contained_file(root, image.get("relative_path"))
        image_png_sha256 = _require_sha256(
            image.get("png_sha256"), field="sample image PNG SHA-256"
        )

        hand = row.get("hand")
        if not isinstance(hand, Mapping):
            raise TypeError("sample hand must be an object")
        hand_index = int(hand.get("hand_index", -1))
        if hand_index < 0:
            raise ValueError("sample hand_index must be non-negative")
        raw_landmarks = hand.get("feature_landmarks")
        if not isinstance(raw_landmarks, list):
            raise TypeError("sample feature_landmarks must be an array")
        landmark_by_index: dict[int, Mapping[str, Any]] = {}
        for raw in raw_landmarks:
            if not isinstance(raw, Mapping):
                raise TypeError("sample landmark must be an object")
            index = int(raw.get("landmark_index", -1))
            if index in landmark_by_index:
                raise ValueError("sample contains a duplicate feature landmark")
            landmark_by_index[index] = raw
        if tuple(landmark_by_index) != available:
            raise ValueError("sample feature landmark order differs from the manifest")
        landmark_xy: list[tuple[float, float]] = []
        for index in selected:
            raw = landmark_by_index[index]
            if raw.get("landmark_name") != HAND_LANDMARK_NAMES[index]:
                raise ValueError("sample landmark name differs from its index")
            x = float(raw.get("x_normalized"))
            y = float(raw.get("y_normalized"))
            if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("sample landmark x/y must be finite normalized coordinates")
            if raw.get("in_frame") is not True:
                raise ValueError("Phase 8 landmark inputs must be in-frame")
            # z_mediapipe_relativeは深度と相関する情報が漏れる可能性があるため、
            # 読み込みも保存もしません。
            landmark_xy.append((x, y))

        target = row.get("target")
        if not isinstance(target, Mapping) or target.get("depth_valid") is not True:
            raise ValueError("sample target must be a valid depth object")
        if (
            int(target.get("landmark_index", -1)) != 8
            or target.get("landmark_name") != HAND_LANDMARK_NAMES[8]
            or target.get("depth_sampling") != "single_pixel"
        ):
            raise ValueError("sample target must be the single-pixel INDEX_FINGER_TIP")
        target_depth_m = float(target.get("z_teacher_m"))
        if not math.isfinite(target_depth_m) or target_depth_m <= 0:
            raise ValueError("sample target depth must be finite and positive")
        augmentation = row.get("augmentation")
        if not isinstance(augmentation, Mapping):
            raise TypeError("sample augmentation must be an object")
        augmentation_source_id = augmentation.get("source_sample_id")
        if augmentation_source_id is not None and str(augmentation_source_id) != source_sample_id:
            raise ValueError("augmentation source_sample_id differs from the sample")
        variant = str(augmentation.get("variant", ""))
        if variant not in {"identity", "hflip"}:
            raise ValueError("sample has an unsupported augmentation variant")
        if split == "validation" and variant != "identity":
            raise ValueError("validation samples must be identity-only")
        samples.append(
            StudentTrainingSample(
                sample_id=sample_id,
                source_sample_id=source_sample_id,
                split=split,  # type: ignore[arg-type]
                image_path=image_path,
                image_png_sha256=image_png_sha256,
                width=width,
                height=height,
                landmark_xy=tuple(landmark_xy),
                target_depth_m=target_depth_m,
                source_sequence_id=source_sequence_id,
                hand_index=hand_index,
                frame_index=frame_index,
                augmentation_variant=variant,  # type: ignore[arg-type]
            )
        )

    train_samples = tuple(sample for sample in samples if sample.split == "train")
    validation_samples = tuple(sample for sample in samples if sample.split == "validation")
    observed_train_sequences = {sample.source_sequence_id for sample in train_samples}
    observed_validation_sequences = {sample.source_sequence_id for sample in validation_samples}
    if (
        observed_train_sequences != train_sequences
        or observed_validation_sequences != validation_sequences
    ):
        raise ValueError("observed sample sequences differ from the split policy")
    if split_unit == "source video chronological frame tail":
        for sequence_id in sorted(train_sequences):
            metadata = chronological_metadata[sequence_id]
            train_identity = [
                sample
                for sample in train_samples
                if sample.source_sequence_id == sequence_id
                and sample.augmentation_variant == "identity"
            ]
            validation_identity = [
                sample
                for sample in validation_samples
                if sample.source_sequence_id == sequence_id
                and sample.augmentation_variant == "identity"
            ]
            validation_frames = [sample.frame_index for sample in validation_identity]
            expected_counts_and_range = (
                int(metadata.get("accepted_train_identity_samples", -1)),
                int(metadata.get("accepted_validation_identity_samples", -1)),
                int(metadata.get("first_accepted_validation_frame", -1)),
                int(metadata.get("last_accepted_validation_frame", -1)),
            )
            observed_counts_and_range = (
                len(train_identity),
                len(validation_identity),
                min(validation_frames),
                max(validation_frames),
            )
            if observed_counts_and_range != expected_counts_and_range:
                raise ValueError(
                    f"chronological accepted-sample metadata is inconsistent: {sequence_id}"
                )
    source_sample_splits: dict[str, str] = {}
    for sample in samples:
        previous = source_sample_splits.setdefault(sample.source_sample_id, sample.split)
        if previous != sample.split:
            raise ValueError("source observation crosses train and validation splits")
    expected_train_ids = artifact_paths["train_split"].read_text(encoding="utf-8").splitlines()
    expected_validation_ids = (
        artifact_paths["validation_split"].read_text(encoding="utf-8").splitlines()
    )
    if [sample.sample_id for sample in train_samples] != expected_train_ids:
        raise ValueError("train split file differs from samples.jsonl order")
    if [sample.sample_id for sample in validation_samples] != expected_validation_ids:
        raise ValueError("validation split file differs from samples.jsonl order")
    if int(counts.get("train_samples_total", -1)) != len(train_samples):
        raise ValueError("manifest train count is inconsistent")
    if int(counts.get("validation_samples_total", -1)) != len(validation_samples):
        raise ValueError("manifest validation count is inconsistent")
    if not train_samples or not validation_samples:
        raise ValueError("both train and validation samples are required")
    return LoadedStudentCorpus(
        manifest_path=manifest_path,
        manifest_sha256=observed,
        manifest=manifest,
        available_landmark_indices=available,
        selected_landmark_indices=selected,
        train_samples=train_samples,
        validation_samples=validation_samples,
        artifact_sha256=artifact_hashes,
    )


# 順序を固定した標本ID一覧のSHA-256を計算します。
def _ordered_sample_ids_sha256(samples: Sequence[StudentTrainingSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# データ分割・系列ごとに教師深度の統計を計算します。
def _identity_depth_statistics(samples: Sequence[StudentTrainingSample]) -> dict[str, Any]:
    values = np.asarray(
        [sample.target_depth_m for sample in samples if sample.augmentation_variant == "identity"],
        dtype=np.float64,
    )
    if values.size == 0:
        return {
            "identity_count": 0,
            "min_m": None,
            "median_m": None,
            "mean_m": None,
            "max_m": None,
        }
    return {
        "identity_count": int(values.size),
        "min_m": float(np.min(values)),
        "median_m": float(np.median(values)),
        "mean_m": float(np.mean(values)),
        "max_m": float(np.max(values)),
    }


# 教師深度の急変を系列内で検出し、設定に応じて該当標本を除外します。
def apply_teacher_spike_filter(
    corpus: LoadedStudentCorpus,
    config: TeacherSpikeFilterConfig,
) -> TeacherSpikeFilterResult:
    """Select an audited cohort using identity-only leave-one-out Hampel decisions."""

    all_samples = (*corpus.train_samples, *corpus.validation_samples)
    views_by_key: dict[tuple[str, int, int], dict[str, StudentTrainingSample]] = {}
    for sample in all_samples:
        key = (sample.source_sequence_id, sample.hand_index, sample.frame_index)
        views = views_by_key.setdefault(key, {})
        if sample.augmentation_variant in views:
            raise ValueError(
                "duplicate augmentation variant for source observation: "
                f"{key}, {sample.augmentation_variant}"
            )
        views[sample.augmentation_variant] = sample

    identities: list[StudentTrainingSample] = []
    for key, views in views_by_key.items():
        identity = views.get("identity")
        if identity is None:
            raise ValueError(f"augmented source observation has no identity view: {key}")
        identities.append(identity)
        for view in views.values():
            if (
                view.source_sample_id != identity.source_sample_id
                or view.split != identity.split
                or view.target_depth_m != identity.target_depth_m
            ):
                raise ValueError(
                    "augmented view differs from identity provenance, split, or teacher depth: "
                    f"{key}"
                )

    grouped: dict[tuple[str, str, int], list[StudentTrainingSample]] = {}
    for identity in identities:
        group_key = (identity.split, identity.source_sequence_id, identity.hand_index)
        grouped.setdefault(group_key, []).append(identity)

    decisions: list[dict[str, Any]] = []
    rejected_keys: set[tuple[str, int, int]] = set()
    for group_key in sorted(grouped):
        ordered = sorted(grouped[group_key], key=lambda sample: sample.frame_index)
        if len({sample.frame_index for sample in ordered}) != len(ordered):
            raise ValueError(f"duplicate identity frame in temporal stream: {group_key}")
        segments: list[list[StudentTrainingSample]] = []
        current: list[StudentTrainingSample] = []
        for identity in ordered:
            if current:
                frame_delta = identity.frame_index - current[-1].frame_index
                if frame_delta <= 0:
                    raise ValueError(f"identity frames are not strictly increasing: {group_key}")
                if frame_delta > config.max_frame_gap:
                    segments.append(current)
                    current = []
            current.append(identity)
        if current:
            segments.append(current)

        for segment_index, segment in enumerate(segments):
            for identity in segment:
                neighbors = [
                    other
                    for other in segment
                    if other.frame_index != identity.frame_index
                    and abs(other.frame_index - identity.frame_index) <= config.frame_radius
                ]
                neighbor_frames = [sample.frame_index for sample in neighbors]
                neighbor_depths = [sample.target_depth_m for sample in neighbors]
                eligible = len(neighbors) >= config.min_neighbors
                local_median_m: float | None = None
                local_mad_m: float | None = None
                scaled_mad_m: float | None = None
                absolute_deviation_m: float | None = None
                relative_ratio: float | None = None
                relative_floor_m: float | None = None
                robust_floor_m: float | None = None
                threshold_m: float | None = None
                candidate_if_enabled = False
                if eligible:
                    depths = np.asarray(neighbor_depths, dtype=np.float64)
                    local_median_m = float(np.median(depths))
                    deviations = np.abs(depths - local_median_m)
                    local_mad_m = float(np.median(deviations))
                    scaled_mad_m = config.mad_scale * local_mad_m
                    absolute_deviation_m = abs(identity.target_depth_m - local_median_m)
                    relative_ratio = max(
                        identity.target_depth_m / local_median_m,
                        local_median_m / identity.target_depth_m,
                    )
                    relative_floor_m = config.relative_floor_fraction * local_median_m
                    robust_floor_m = config.mad_multiplier * scaled_mad_m
                    threshold_m = max(
                        config.absolute_floor_m,
                        relative_floor_m,
                        robust_floor_m,
                    )
                    candidate_if_enabled = absolute_deviation_m > threshold_m

                rejected = config.enabled and candidate_if_enabled
                key = (
                    identity.source_sequence_id,
                    identity.hand_index,
                    identity.frame_index,
                )
                if rejected:
                    rejected_keys.add(key)
                if not config.enabled:
                    reason = "filter_disabled"
                elif not eligible:
                    reason = "insufficient_contiguous_neighbors"
                elif rejected:
                    reason = "leave_one_out_hampel_excursion"
                else:
                    reason = "within_threshold"
                removed_sample_ids = (
                    [sample.sample_id for sample in views_by_key[key].values()] if rejected else []
                )
                decisions.append(
                    {
                        "source_sample_id": identity.source_sample_id,
                        "identity_sample_id": identity.sample_id,
                        "source_sequence_id": identity.source_sequence_id,
                        "split": identity.split,
                        "hand_index": identity.hand_index,
                        "frame_index": identity.frame_index,
                        "segment_index": segment_index,
                        "target_depth_m": identity.target_depth_m,
                        "neighbor_frame_indices": neighbor_frames,
                        "neighbor_depth_m": neighbor_depths,
                        "neighbor_count": len(neighbors),
                        "eligible": eligible,
                        "local_median_m": local_median_m,
                        "local_mad_m": local_mad_m,
                        "scaled_mad_m": scaled_mad_m,
                        "absolute_deviation_m": absolute_deviation_m,
                        "relative_ratio": relative_ratio,
                        "absolute_floor_m": config.absolute_floor_m,
                        "relative_floor_m": relative_floor_m,
                        "robust_floor_m": robust_floor_m,
                        "threshold_m": threshold_m,
                        "candidate_if_enabled": candidate_if_enabled,
                        "rejected": rejected,
                        "reason": reason,
                        "removed_sample_ids": removed_sample_ids,
                    }
                )

    # 外れ値判定後に保持する標本一覧を返します。
    def retained(samples: Sequence[StudentTrainingSample]) -> tuple[StudentTrainingSample, ...]:
        return tuple(
            sample
            for sample in samples
            if (sample.source_sequence_id, sample.hand_index, sample.frame_index)
            not in rejected_keys
        )

    train_samples = retained(corpus.train_samples)
    validation_samples = retained(corpus.validation_samples)
    if not train_samples or not validation_samples:
        raise ValueError("teacher spike filtering removed an entire train or validation split")

    sequence_summary: dict[str, dict[str, int]] = {}
    for decision in decisions:
        summary = sequence_summary.setdefault(
            str(decision["source_sequence_id"]),
            {
                "identity_before": 0,
                "identity_rejected": 0,
                "views_removed": 0,
                "identity_after": 0,
            },
        )
        summary["identity_before"] += 1
        if decision["rejected"]:
            summary["identity_rejected"] += 1
            summary["views_removed"] += len(decision["removed_sample_ids"])
        else:
            summary["identity_after"] += 1

    rejected_decisions = [decision for decision in decisions if decision["rejected"]]
    report: dict[str, Any] = {
        "format": TEACHER_SPIKE_FILTER_FORMAT,
        "format_version": TEACHER_SPIKE_FILTER_FORMAT_VERSION,
        "algorithm": {
            "id": "leave_one_out_hampel_v1",
            "config": config.as_dict(),
            "decision_source": "identity observations only",
            "grouping": ["split", "source_sequence_id", "hand_index"],
            "gap_policy": (
                "sort by frame_index and start a new contiguous segment when frame delta "
                "exceeds max_frame_gap"
            ),
            "boundary_policy": (
                "truncated one-sided windows are eligible when min_neighbors is met"
            ),
            "candidate_excluded_from_own_window": True,
            "decisions_computed_simultaneously_from_unfiltered_identity_series": True,
            "threshold_formula": (
                "max(absolute_floor_m, relative_floor_fraction * local_median_m, "
                "mad_multiplier * mad_scale * local_MAD_m)"
            ),
            "comparison": "absolute_deviation_m > threshold_m",
            "hflip_policy": (
                "never participates in temporal statistics; inherits identity decision "
                "by (source_sequence_id, hand_index, frame_index)"
            ),
            "offline_noncausal_data_curation": True,
            "rgb_used": False,
            "landmark_xy_used": False,
            "z_mediapipe_relative_used": False,
            "prediction_used": False,
        },
        "source_dataset_manifest_sha256": corpus.manifest_sha256,
        "counts": {
            "train_samples_before": len(corpus.train_samples),
            "train_samples_after": len(train_samples),
            "validation_samples_before": len(corpus.validation_samples),
            "validation_samples_after": len(validation_samples),
            "identity_decisions": len(decisions),
            "identity_rejected": len(rejected_decisions),
            "views_removed_total": len(all_samples) - len(train_samples) - len(validation_samples),
        },
        "identity_depth_statistics_before": {
            "train": _identity_depth_statistics(corpus.train_samples),
            "validation": _identity_depth_statistics(corpus.validation_samples),
        },
        "identity_depth_statistics_after": {
            "train": _identity_depth_statistics(train_samples),
            "validation": _identity_depth_statistics(validation_samples),
        },
        "sequence_summary": sequence_summary,
        "rejected_observations": rejected_decisions,
        "included_train_ids_sha256": _ordered_sample_ids_sha256(train_samples),
        "included_validation_ids_sha256": _ordered_sample_ids_sha256(validation_samples),
    }
    return TeacherSpikeFilterResult(
        config=config,
        train_samples=train_samples,
        validation_samples=validation_samples,
        decisions=tuple(decisions),
        report=report,
        included_train_ids_sha256=str(report["included_train_ids_sha256"]),
        included_validation_ids_sha256=str(report["included_validation_ids_sha256"]),
    )


# 学習・評価ループへ標本を渡すPyTorchデータセットです。
class FingertipStudentDataset(Dataset[dict[str, Any]]):
    """Direct-resize RGB and x/y-only hand landmarks for one fixed split."""

    # 必要な引数を検証し、インスタンスの状態を初期化します。
    def __init__(
        self,
        samples: Sequence[StudentTrainingSample],
        *,
        image_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        preload_images: bool,
        verify_png_sha256: bool,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("student Dataset must not be empty")
        self.samples = tuple(samples)
        self.image_size = int(image_size)
        self.image_mean = torch.tensor(tuple(image_mean), dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(tuple(image_std), dtype=torch.float32).view(3, 1, 1)
        self.verify_png_sha256 = bool(verify_png_sha256)
        self._image_cache: np.ndarray | None = None
        if preload_images:
            self._image_cache = np.empty(
                (len(self.samples), 3, self.image_size, self.image_size), dtype=np.uint8
            )
            for index, sample in enumerate(self.samples):
                self._image_cache[index] = self._load_image(sample)
                if progress is not None and (index + 1) % 500 == 0:
                    progress(f"preloaded {index + 1}/{len(self.samples)} images")

    # 標本画像を読み込み、モデル入力サイズと画素範囲に整えます。
    def _load_image(self, sample: StudentTrainingSample) -> np.ndarray:
        encoded = sample.image_path.read_bytes()
        if self.verify_png_sha256:
            observed = hashlib.sha256(encoded).hexdigest()
            if observed != sample.image_png_sha256:
                raise ValueError(f"sample PNG SHA-256 mismatch: {sample.sample_id}")
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (sample.height, sample.width, 3):
            raise ValueError(f"sample image shape differs from metadata: {sample.sample_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        return np.ascontiguousarray(resized.transpose(2, 0, 1))

    # データセットに含まれる標本の件数を返します。
    def __len__(self) -> int:
        return len(self.samples)

    # 指定された標本の画像・入力特徴・教師値を返します。
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_uint8 = (
            self._image_cache[index] if self._image_cache is not None else self._load_image(sample)
        )
        image = torch.from_numpy(image_uint8).to(dtype=torch.float32).div_(255.0)
        image = image.sub(self.image_mean).div(self.image_std)
        # X・Y座標を[-1, 1]へ変換します。MediaPipeの相対Zは入力に含めません。
        landmark_xy = torch.tensor(sample.landmark_xy, dtype=torch.float32)
        landmark_xy = landmark_xy.mul(2.0).sub(1.0)
        return {
            "pixel_values": image,
            "landmark_coordinates": landmark_xy,
            "target_depth_m": torch.tensor(sample.target_depth_m, dtype=torch.float32),
            "sample_id": sample.sample_id,
            "source_sequence_id": sample.source_sequence_id,
            "frame_index": sample.frame_index,
            "augmentation_variant": sample.augmentation_variant,
        }


# 予測値と教師値から回帰損失、絶対誤差などの指標を計算します。
def regression_metrics(prediction_m: np.ndarray, target_m: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    if prediction.ndim != 1 or target.ndim != 1 or prediction.shape != target.shape:
        raise ValueError("predictions and targets must be same-length one-dimensional arrays")
    if (
        prediction.size == 0
        or not np.all(np.isfinite(prediction))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("predictions and targets must be non-empty and finite")
    error = prediction - target
    absolute = np.abs(error)
    mse = float(np.mean(np.square(error)))
    correlation: float | None = None
    if prediction.size > 1 and float(np.std(prediction)) > 0 and float(np.std(target)) > 0:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "count": int(prediction.size),
        "mse_m2": mse,
        "rmse_m": math.sqrt(mse),
        "rmse_cm": 100.0 * math.sqrt(mse),
        "mae_m": float(np.mean(absolute)),
        "mae_cm": 100.0 * float(np.mean(absolute)),
        "median_absolute_error_m": float(np.median(absolute)),
        "median_absolute_error_cm": 100.0 * float(np.median(absolute)),
        "p95_absolute_error_m": float(np.percentile(absolute, 95.0)),
        "p95_absolute_error_cm": 100.0 * float(np.percentile(absolute, 95.0)),
        "bias_m": float(np.mean(error)),
        "bias_cm": 100.0 * float(np.mean(error)),
        "max_absolute_error_m": float(np.max(absolute)),
        "negative_prediction_count": int(np.count_nonzero(prediction < 0.0)),
        "pearson_r": correlation,
    }


# 予測と教師の軌跡について、フレーム間移動量の誤差を集計します。
def _trajectory_delta_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    absolute_errors: list[float] = []
    pair_count = 0
    streams: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["source_sequence_id"]), str(row["augmentation_variant"]))
        streams.setdefault(key, []).append(row)
    for stream_rows in streams.values():
        ordered = sorted(stream_rows, key=lambda row: int(row["frame_index"]))
        for previous, current in pairwise(ordered):
            consecutive = int(current["frame_index"]) == int(previous["frame_index"]) + 1
            if not consecutive:
                continue
            predicted_delta = float(current["prediction_m"]) - float(previous["prediction_m"])
            target_delta = float(current["target_m"]) - float(previous["target_m"])
            absolute_errors.append(abs(predicted_delta - target_delta))
            pair_count += 1
    if not absolute_errors:
        return {"consecutive_pair_count": 0, "delta_mae_m": None, "delta_mae_cm": None}
    mean_error = float(np.mean(np.asarray(absolute_errors, dtype=np.float64)))
    return {
        "consecutive_pair_count": pair_count,
        "delta_mae_m": mean_error,
        "delta_mae_cm": 100.0 * mean_error,
    }


# Python・NumPy・PyTorchなどの乱数シードを設定します。
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# モデル重み辞書の内容を決定的にシリアライズしてSHA-256を計算します。
def _state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# 設定に従って学習対象パラメーターをまとめ、最適化器を作成します。
def _build_optimizer(
    model: FingertipDepthStudent,
    config: StudentTrainingConfig,
) -> torch.optim.AdamW:
    encoder_parameter_ids = {id(parameter) for parameter in model.image_encoder.parameters()}
    groups: dict[tuple[str, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = "encoder" if id(parameter) in encoder_parameter_ids else "new_modules"
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        decay = 0.0 if no_decay else config.weight_decay
        groups.setdefault((group_name, decay), []).append(parameter)
    parameter_groups: list[dict[str, Any]] = []
    for (group_name, decay), parameters in groups.items():
        learning_rate = (
            config.encoder_learning_rate if group_name == "encoder" else config.head_learning_rate
        )
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": decay,
                "group_name": f"{group_name}_{'decay' if decay else 'no_decay'}",
            }
        )
    return torch.optim.AdamW(parameter_groups)


# ウォームアップ後に学習率をコサイン減衰させるスケジューラーを作成します。
def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = round(total_steps * warmup_fraction)

    # 比率と基準値から適用する件数倍率を計算します。
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


# デバイスと設定に応じて自動混合精度を有効にするか判定します。
def _autocast_enabled(config: StudentTrainingConfig, device: torch.device) -> bool:
    return config.precision == "bfloat16" and device.type == "cuda"


# 学習データを一巡し、損失に基づいてモデルの重みを更新します。
def _train_one_epoch(
    model: FingertipDepthStudent,
    loader: DataLoader[dict[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: StudentTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    if config.freeze_image_encoder:
        model.image_encoder.eval()
    squared_error_sum = 0.0
    sample_count = 0
    gradient_norm_sum = 0.0
    step_count = 0
    for batch in loader:
        images = batch["pixel_values"].to(device=device, non_blocking=True)
        landmarks = batch["landmark_coordinates"].to(device=device, non_blocking=True)
        targets = batch["target_depth_m"].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=_autocast_enabled(config, device),
        ):
            predictions = model(images, landmarks)
        loss = F.mse_loss(predictions.float(), targets.float())
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
            config.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()
        batch_size = int(targets.shape[0])
        squared_error_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
        gradient_norm_sum += float(gradient_norm.detach().cpu())
        step_count += 1
    return {
        "mse_m2": squared_error_sum / sample_count,
        "mean_preclip_gradient_norm": gradient_norm_sum / max(step_count, 1),
    }


# 評価データを推論し、損失と深度推定指標を計算します。
def _evaluate_model(
    model: FingertipDepthStudent,
    loader: DataLoader[dict[str, Any]],
    *,
    config: StudentTrainingConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["pixel_values"].to(device=device, non_blocking=True)
            landmarks = batch["landmark_coordinates"].to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=_autocast_enabled(config, device),
            ):
                predictions = model(images, landmarks)
            prediction_values = predictions.float().cpu().numpy()
            target_values = batch["target_depth_m"].numpy()
            frame_indices = batch["frame_index"].numpy()
            for index in range(len(prediction_values)):
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "source_sequence_id": str(batch["source_sequence_id"][index]),
                        "frame_index": int(frame_indices[index]),
                        "augmentation_variant": str(batch["augmentation_variant"][index]),
                        "target_m": float(target_values[index]),
                        "prediction_m": float(prediction_values[index]),
                    }
                )
    predictions = np.asarray([row["prediction_m"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["target_m"] for row in rows], dtype=np.float64)
    metrics = regression_metrics(predictions, targets)
    metrics["consecutive_frame_delta"] = _trajectory_delta_metrics(rows)
    return metrics, rows


# レコード群をJSON Linesとして保存します。
def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            target.write("\n")


# スパイク判定、除外標本、適用条件を監査記録に保存します。
def _write_teacher_spike_filter_artifacts(
    output_dir: Path,
    result: TeacherSpikeFilterResult,
) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions_path = output_dir / "teacher_spike_filter_decisions.jsonl"
    train_ids_path = output_dir / "included_train.txt"
    validation_ids_path = output_dir / "included_validation.txt"
    report_path = output_dir / "teacher_spike_filter.json"

    _write_jsonl(decisions_path, result.decisions)
    train_ids_path.write_text(
        "".join(f"{sample.sample_id}\n" for sample in result.train_samples),
        encoding="utf-8",
    )
    validation_ids_path.write_text(
        "".join(f"{sample.sample_id}\n" for sample in result.validation_samples),
        encoding="utf-8",
    )
    if sha256_file(train_ids_path) != result.included_train_ids_sha256:
        raise AssertionError("included train sample-ID hash differs from the filter result")
    if sha256_file(validation_ids_path) != result.included_validation_ids_sha256:
        raise AssertionError("included validation sample-ID hash differs from the filter result")

    report = dict(result.report)
    report["artifacts"] = {
        "identity_decisions": {
            "relative_path": decisions_path.name,
            "sha256": sha256_file(decisions_path),
        },
        "included_train": {
            "relative_path": train_ids_path.name,
            "sha256": sha256_file(train_ids_path),
        },
        "included_validation": {
            "relative_path": validation_ids_path.name,
            "sha256": sha256_file(validation_ids_path),
        },
    }
    write_json(report_path, report)
    artifacts = {
        "teacher_spike_filter": {
            "relative_path": report_path.name,
            "sha256": sha256_file(report_path),
        },
        **dict(report["artifacts"]),
    }
    return report, artifacts


# 標本ID、教師値、予測値を対応づけたCSVを保存します。
def _write_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "sample_id",
        "source_sequence_id",
        "frame_index",
        "augmentation_variant",
        "target_m",
        "prediction_m",
        "error_m",
        "absolute_error_m",
    )
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            error = float(row["prediction_m"]) - float(row["target_m"])
            writer.writerow(
                {
                    **dict(row),
                    "error_m": error,
                    "absolute_error_m": abs(error),
                }
            )


# モデル重み、設定、最良指標を再開可能なチェックポイントに保存します。
def _save_checkpoint(
    path: Path,
    *,
    model: FingertipDepthStudent,
    epoch: int,
    validation_metrics: Mapping[str, Any],
    corpus: LoadedStudentCorpus,
    model_config: StudentModelConfig,
    training_config: StudentTrainingConfig,
    spike_filter_config: TeacherSpikeFilterConfig,
    selection_provenance: Mapping[str, Any],
    initial_encoder_state_sha256: str,
) -> None:
    payload = {
        "format": "fingertip-depth-student-checkpoint",
        "format_version": STUDENT_CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "dataset_manifest_sha256": corpus.manifest_sha256,
        "model_config": model_config.as_dict(),
        "training_config": training_config.as_dict(),
        "teacher_spike_filter_config": spike_filter_config.as_dict(),
        "data_selection": _json_safe(selection_provenance),
        "initial_image_encoder_state_sha256": initial_encoder_state_sha256,
        "validation_metrics": dict(validation_metrics),
        "model_state_dict": model.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


# 任意の設定値をJSONで表現できる標準型へ変換します。
def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


# 関係するPythonソースのSHA-256を集めて再現性を記録します。
def _implementation_hashes() -> dict[str, str]:
    project_dir = Path(__file__).resolve().parents[2]
    paths = {
        "student_model.py": project_dir / "src" / "fingertip_depth" / "student_model.py",
        "student_training.py": Path(__file__),
        "student_dataset.py": project_dir / "src" / "fingertip_depth" / "student_dataset.py",
        "constants.py": project_dir / "src" / "fingertip_depth" / "constants.py",
        "artifacts.py": project_dir / "src" / "fingertip_depth" / "artifacts.py",
        "video_cache.py": project_dir / "src" / "fingertip_depth" / "video_cache.py",
        "train_student_transformer.py": project_dir / "scripts" / "train_student_transformer.py",
        "pyproject.toml": project_dir / "pyproject.toml",
        "uv.lock": project_dir / "uv.lock",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


# Python・ライブラリ・実行デバイスなど実行環境を記録します。
def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    cuda: dict[str, Any] | None = None
    if device.type == "cuda":
        cuda = {
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "bfloat16_supported": torch.cuda.is_bf16_supported(),
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "device": str(device),
        "cuda": cuda,
    }


# ランドマーク種別トークンの初期値と学習後の変化を集計します。
def _type_token_change_report(
    initial_tokens: Sequence[torch.Tensor],
    model: FingertipDepthStudent,
) -> dict[str, Any]:
    selected = frozenset(model.config.landmark_indices)
    rows: list[dict[str, Any]] = []
    all_frozen_unchanged = True
    all_selected_changed = True
    for index, (initial, current_parameter) in enumerate(
        zip(initial_tokens, model.landmark_type_tokens.tokens, strict=True)
    ):
        current = current_parameter.detach().cpu()
        exactly_equal = torch.equal(initial, current)
        change_norm = float(torch.linalg.vector_norm(current - initial))
        is_selected = index in selected
        if is_selected:
            all_selected_changed &= not exactly_equal
        else:
            all_frozen_unchanged &= exactly_equal
        rows.append(
            {
                "landmark_index": index,
                "landmark_name": HAND_LANDMARK_NAMES[index],
                "selected_for_input": is_selected,
                "requires_grad": bool(current_parameter.requires_grad),
                "exactly_equal_to_initial": exactly_equal,
                "l2_change": change_norm,
            }
        )
    if not all_frozen_unchanged:
        raise AssertionError("an unselected landmark type token changed during training")
    if not all_selected_changed:
        raise AssertionError("at least one selected landmark type token did not learn")
    return {
        "all_21_tokens_present": len(rows) == len(HAND_LANDMARK_NAMES),
        "selected_tokens_all_changed": all_selected_changed,
        "unselected_tokens_all_bit_exact_unchanged": all_frozen_unchanged,
        "tokens": rows,
    }


# コーパスを使って生徒モデルを学習し、評価結果とチェックポイントを保存します。
def train_student_transformer(
    *,
    dataset_manifest_path: Path,
    expected_dataset_manifest_sha256: str,
    output_dir: Path,
    model_config: StudentModelConfig,
    training_config: StudentTrainingConfig,
    device_name: str,
    spike_filter_config: TeacherSpikeFilterConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train and evaluate the Phase 8 model on an audited fixed split."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {output_dir}")
    if spike_filter_config is None:
        spike_filter_config = TeacherSpikeFilterConfig()
    corpus = load_student_corpus(
        dataset_manifest_path,
        expected_manifest_sha256=expected_dataset_manifest_sha256,
        landmark_indices=model_config.landmark_indices,
    )
    selection = apply_teacher_spike_filter(corpus, spike_filter_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_report, filter_artifacts = _write_teacher_spike_filter_artifacts(output_dir, selection)
    selection_provenance = {
        "source_dataset_manifest_sha256": corpus.manifest_sha256,
        "teacher_spike_filter_config": spike_filter_config.as_dict(),
        "teacher_spike_filter_report_sha256": filter_artifacts["teacher_spike_filter"]["sha256"],
        "identity_decisions_sha256": filter_artifacts["identity_decisions"]["sha256"],
        "included_train_ids_sha256": selection.included_train_ids_sha256,
        "included_validation_ids_sha256": selection.included_validation_ids_sha256,
        "checkpoint_selection_scope": "filtered_validation",
        "raw_validation_scope": "post-hoc diagnostic only; not used for selection",
    }
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if (
        training_config.precision == "bfloat16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 precision was requested but is not supported")
    _seed_everything(training_config.seed)
    torch.set_float32_matmul_precision("high")

    train_targets = np.asarray(
        [sample.target_depth_m for sample in selection.train_samples], dtype=np.float64
    )
    validation_targets = np.asarray(
        [sample.target_depth_m for sample in selection.validation_samples], dtype=np.float64
    )
    raw_validation_targets = np.asarray(
        [sample.target_depth_m for sample in corpus.validation_samples], dtype=np.float64
    )
    train_target_mean = float(np.mean(train_targets))
    train_target_median = float(np.median(train_targets))

    if progress is not None:
        progress(f"loading image encoder {model_config.image_encoder_name}")
    model = FingertipDepthStudent(
        model_config,
        initial_depth_bias_m=train_target_mean,
    )
    pretrained_cfg = _json_safe(getattr(model.image_encoder, "pretrained_cfg", {}))
    expected_input_size = tuple(
        int(value)
        for value in getattr(model.image_encoder, "pretrained_cfg", {}).get(
            "input_size", (3, training_config.image_size, training_config.image_size)
        )
    )
    if expected_input_size != (3, training_config.image_size, training_config.image_size):
        raise ValueError(
            f"training image size differs from encoder input size: {expected_input_size}"
        )
    encoder_mean = tuple(float(value) for value in pretrained_cfg.get("mean", []))
    encoder_std = tuple(float(value) for value in pretrained_cfg.get("std", []))
    if encoder_mean and encoder_mean != training_config.image_mean:
        raise ValueError("training image mean differs from the pretrained encoder")
    if encoder_std and encoder_std != training_config.image_std:
        raise ValueError("training image std differs from the pretrained encoder")
    if training_config.freeze_image_encoder:
        model.image_encoder.requires_grad_(False)
    initial_encoder_state_sha256 = _state_dict_sha256(model.image_encoder)
    initial_type_tokens = [
        parameter.detach().cpu().clone() for parameter in model.landmark_type_tokens.tokens
    ]

    if progress is not None:
        progress("preloading and verifying train images")
    train_dataset = FingertipStudentDataset(
        selection.train_samples,
        image_size=training_config.image_size,
        image_mean=training_config.image_mean,
        image_std=training_config.image_std,
        preload_images=training_config.preload_images,
        verify_png_sha256=training_config.verify_image_png_sha256,
        progress=progress,
    )
    if progress is not None:
        progress("preloading and verifying validation images")
    validation_dataset = FingertipStudentDataset(
        selection.validation_samples,
        image_size=training_config.image_size,
        image_mean=training_config.image_mean,
        image_std=training_config.image_std,
        preload_images=training_config.preload_images,
        verify_png_sha256=training_config.verify_image_png_sha256,
        progress=progress,
    )

    generator = torch.Generator().manual_seed(training_config.seed)
    loader_common = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": training_config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_common,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_common)
    train_evaluation_loader = DataLoader(train_dataset, shuffle=False, **loader_common)

    model.to(device)
    optimizer = _build_optimizer(model, training_config)
    total_steps = training_config.epochs * len(train_loader)
    scheduler = _cosine_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=training_config.warmup_fraction,
    )
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    history_path = output_dir / "history.jsonl"
    filtered_validation_predictions_path = output_dir / "validation_predictions_filtered.csv"
    raw_validation_predictions_path = output_dir / "validation_predictions_raw.csv"

    filtered_validation_baseline = {
        "constant_train_mean": regression_metrics(
            np.full_like(validation_targets, train_target_mean), validation_targets
        ),
        "constant_train_median": regression_metrics(
            np.full_like(validation_targets, train_target_median), validation_targets
        ),
    }
    raw_validation_baseline = {
        "constant_train_mean": regression_metrics(
            np.full_like(raw_validation_targets, train_target_mean), raw_validation_targets
        ),
        "constant_train_median": regression_metrics(
            np.full_like(raw_validation_targets, train_target_median), raw_validation_targets
        ),
    }
    history: list[dict[str, Any]] = []
    best_validation_mse = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()
    for epoch in range(1, training_config.epochs + 1):
        epoch_start = time.perf_counter()
        train_step_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=training_config,
            device=device,
        )
        validation_metrics, _ = _evaluate_model(
            model,
            validation_loader,
            config=training_config,
            device=device,
        )
        record = {
            "epoch": epoch,
            "train": train_step_metrics,
            "validation": validation_metrics,
            "learning_rates": {
                str(group.get("group_name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            },
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        _write_jsonl(history_path, history)
        current_mse = float(validation_metrics["mse_m2"])
        improved = current_mse < best_validation_mse
        if improved:
            best_validation_mse = current_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                epoch=epoch,
                validation_metrics=validation_metrics,
                corpus=corpus,
                model_config=model_config,
                spike_filter_config=spike_filter_config,
                selection_provenance=selection_provenance,
                training_config=training_config,
                initial_encoder_state_sha256=initial_encoder_state_sha256,
            )
        else:
            epochs_without_improvement += 1
        if progress is not None:
            progress(
                f"epoch {epoch}/{training_config.epochs}: "
                f"train_mse={train_step_metrics['mse_m2']:.8f} "
                f"val_mse={current_mse:.8f} val_mae_cm={validation_metrics['mae_cm']:.3f} "
                f"best={'yes' if improved else 'no'}"
            )
        if (
            training_config.early_stopping_patience > 0
            and epochs_without_improvement >= training_config.early_stopping_patience
        ):
            if progress is not None:
                progress(f"early stopping at epoch {epoch}")
            break

    if not best_checkpoint_path.is_file():
        raise AssertionError("training did not produce a best checkpoint")
    _save_checkpoint(
        last_checkpoint_path,
        model=model,
        epoch=int(history[-1]["epoch"]),
        validation_metrics=history[-1]["validation"],
        corpus=corpus,
        model_config=model_config,
        spike_filter_config=spike_filter_config,
        selection_provenance=selection_provenance,
        training_config=training_config,
        initial_encoder_state_sha256=initial_encoder_state_sha256,
    )
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_filtered_validation_metrics, filtered_validation_rows = _evaluate_model(
        model,
        validation_loader,
        config=training_config,
        device=device,
    )
    train_metrics, _ = _evaluate_model(
        model,
        train_evaluation_loader,
        config=training_config,
        device=device,
    )
    if progress is not None:
        progress("loading raw validation images for post-hoc diagnostic")
    raw_validation_dataset = FingertipStudentDataset(
        corpus.validation_samples,
        image_size=training_config.image_size,
        image_mean=training_config.image_mean,
        image_std=training_config.image_std,
        preload_images=training_config.preload_images,
        verify_png_sha256=training_config.verify_image_png_sha256,
        progress=progress,
    )
    raw_validation_loader = DataLoader(raw_validation_dataset, shuffle=False, **loader_common)
    raw_validation_metrics, raw_validation_rows = _evaluate_model(
        model, raw_validation_loader, config=training_config, device=device
    )
    _write_predictions_csv(filtered_validation_predictions_path, filtered_validation_rows)
    _write_predictions_csv(raw_validation_predictions_path, raw_validation_rows)
    token_changes = _type_token_change_report(initial_type_tokens, model)
    best_encoder_state_sha256 = _state_dict_sha256(model.image_encoder)
    if training_config.freeze_image_encoder:
        if best_encoder_state_sha256 != initial_encoder_state_sha256:
            raise AssertionError("frozen image encoder changed during training")
    elif best_encoder_state_sha256 == initial_encoder_state_sha256:
        raise AssertionError("trainable image encoder did not change during training")

    duration_seconds = time.perf_counter() - start_time
    run_manifest: dict[str, Any] = {
        "format": TRAINING_RUN_FORMAT,
        "format_version": TRAINING_RUN_FORMAT_VERSION,
        "task": {
            "type": "single-frame fingertip optical-axis depth regression",
            "target": "Depth Pro pseudo-label z_teacher_m",
            "target_unit": "metre",
            "output": "one raw scalar depth_logit_m; no sigmoid, softplus, or clipping",
            "loss": "mean squared error in metre squared",
            "target_standardization": False,
            "ground_truth": False,
        },
        "dataset": {
            "manifest_path": str(corpus.manifest_path),
            "manifest_sha256": corpus.manifest_sha256,
            "artifact_sha256": dict(corpus.artifact_sha256),
            "raw_train_samples": len(corpus.train_samples),
            "raw_validation_samples": len(corpus.validation_samples),
            "selected_train_samples": len(selection.train_samples),
            "selected_validation_samples": len(selection.validation_samples),
            "teacher_spike_filter": filter_report,
            "selection_provenance": selection_provenance,
            "split_policy": _json_safe(corpus.manifest["split_policy"]),
        },
        "inputs": {
            "rgb": {
                "resize": [training_config.image_size, training_config.image_size],
                "resize_policy": "direct bicubic resize; no crop or letterbox",
                "color_order": "RGB",
                "mean": list(training_config.image_mean),
                "std": list(training_config.image_std),
            },
            "landmarks": {
                "available_indices_in_dataset": list(corpus.available_landmark_indices),
                "selected_indices": list(corpus.selected_landmark_indices),
                "selected_names": [
                    HAND_LANDMARK_NAMES[index] for index in corpus.selected_landmark_indices
                ],
                "numeric_features": ["2*x_normalized-1", "2*y_normalized-1"],
                "z_mediapipe_relative_used": False,
                "z_exclusion_reason": (
                    "relative z can provide a depth-correlated shortcut/leakage signal"
                ),
                "all_21_type_tokens_instantiated": True,
                "only_selected_type_tokens_trainable": True,
            },
            "camera_intrinsics_used": False,
            "handedness_used": False,
        },
        "model": {
            "config": model_config.as_dict(),
            "parameter_counts": model.parameter_counts(),
            "image_encoder_pretrained_config": pretrained_cfg,
            "image_encoder_initial_state_sha256": initial_encoder_state_sha256,
            "image_encoder_best_state_sha256": best_encoder_state_sha256,
            "landmark_type_token_changes": token_changes,
        },
        "optimization": training_config.as_dict(),
        "target_statistics": {
            "filtered_train_selection": {
                "count": int(train_targets.size),
                "mean_m": train_target_mean,
                "median_m": train_target_median,
                "min_m": float(np.min(train_targets)),
                "max_m": float(np.max(train_targets)),
            },
            "filtered_validation_selection": {
                "count": int(validation_targets.size),
                "min_m": float(np.min(validation_targets)),
                "max_m": float(np.max(validation_targets)),
            },
            "raw_validation_posthoc_diagnostic": {
                "count": int(raw_validation_targets.size),
                "min_m": float(np.min(raw_validation_targets)),
                "max_m": float(np.max(raw_validation_targets)),
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
            },
        },
        "results": {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "checkpoint_selection": {
                "cohort": "filtered_validation",
                "metric": "mse_m2",
            },
            "best_validation": best_filtered_validation_metrics,
            "best_validation_filtered": best_filtered_validation_metrics,
            "raw_validation_posthoc_diagnostic": raw_validation_metrics,
            "train_at_best_checkpoint": train_metrics,
            "validation_baselines": filtered_validation_baseline,
            "raw_validation_posthoc_baselines": raw_validation_baseline,
            "duration_seconds": duration_seconds,
        },
        "artifacts": {
            "best_checkpoint": {
                "relative_path": best_checkpoint_path.name,
                "sha256": sha256_file(best_checkpoint_path),
            },
            "last_checkpoint": {
                "relative_path": last_checkpoint_path.name,
                "sha256": sha256_file(last_checkpoint_path),
            },
            "history": {
                "relative_path": history_path.name,
                "sha256": sha256_file(history_path),
            },
            "validation_predictions_filtered": {
                "relative_path": filtered_validation_predictions_path.name,
                "sha256": sha256_file(filtered_validation_predictions_path),
            },
            "validation_predictions_raw": {
                "relative_path": raw_validation_predictions_path.name,
                "sha256": sha256_file(raw_validation_predictions_path),
            },
            **filter_artifacts,
        },
        "provenance": {
            "implementation_sha256": _implementation_hashes(),
            "data_selection": selection_provenance,
            "runtime": _runtime_metadata(device),
            "seed": training_config.seed,
            "source_images_preloaded": training_config.preload_images,
            "source_png_sha256_verified": training_config.verify_image_png_sha256,
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    return run_manifest


__all__ = [
    "DEFAULT_DATASET_MANIFEST_SHA256",
    "FingertipStudentDataset",
    "LoadedStudentCorpus",
    "StudentTrainingConfig",
    "StudentTrainingSample",
    "TeacherSpikeFilterConfig",
    "TeacherSpikeFilterResult",
    "apply_teacher_spike_filter",
    "load_student_corpus",
    "regression_metrics",
    "train_student_transformer",
]
