'複数系列の監査済み疑似ラベルを統合して生徒モデル用データセットを作ります。訓練データだけを左右反転し、画像・座標・カメラ情報を整合させます。\n\n左右反転では教師推論を再実行しません。光軸方向の深度は保ち、画像上のX座標とカメラ情報を反転後の幾何に合わせます。'

from __future__ import annotations

import copy
import csv
import errno
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .artifacts import write_json
from .coordinates import normalized_to_pixel
from .geometry import CAMERA_COORDINATE_CONVENTION
from .pseudo_labels import (
    DEPTH_PRO_TEACHER_CONDITION_ID,
    PSEUDO_LABEL_DATASET_FORMAT,
    PSEUDO_LABEL_DATASET_FORMAT_VERSION,
)
from .video_cache import pixel_hash_sequence_sha256, pixel_sha256, sha256_file

STUDENT_DATASET_FORMAT = "fingertip-depth-student-dataset"
STUDENT_DATASET_FORMAT_VERSION = 1
STUDENT_SAMPLE_SCHEMA_VERSION = 1

_ALLOWED_SPLITS = frozenset({"train", "validation"})
_SHA256_LENGTH = 64
_GEOMETRY_REL_TOL = 1e-9
_GEOMETRY_ABS_TOL = 1e-12
_EXPECTED_FILTERING = {
    "depth_extraction": "single pixel only; Phase 3 ROI comparison skipped",
    "temporal_filtering": "none; Phase 4 temporal filtering skipped",
    "label_clipping": "none",
    "structural_checks_only": True,
}
_EXPECTED_CAMERA_COORDINATE_SYSTEM = {
    "convention": CAMERA_COORDINATE_CONVENTION,
    "x_formula": "(u_px - cx_px) * z_m / fx_px",
    "y_formula": "(v_px - cy_px) * z_m / fy_px",
    "z_formula": "teacher optical-axis depth in metres",
}
_STABLE_TEACHER_FIELDS = (
    "condition_id",
    "camera_mode",
    "depth_sampling",
    "metric_depth_unit",
    "depth_semantics_assumption",
    "fitted_scale_or_offset_applied",
    "confidence_available",
)
_STABLE_MODEL_FIELDS = (
    "backend",
    "model",
    "source_repository",
    "source_commit",
    "checkpoint_repository",
    "checkpoint_revision",
    "checkpoint_filename",
    "checkpoint_sha256",
    "precision",
    "depth_unit",
)


# 比率から必要件数を計算し、端数を切り上げます。
def _ceil_fractional_count(total: int, fraction: float) -> int:
    """Return ceil(total * fraction) without binary-float boundary drift."""

    if total < 0:
        raise ValueError("total must be non-negative")
    rational = Fraction(str(fraction))
    numerator = total * rational.numerator
    return (numerator + rational.denominator - 1) // rational.denominator


# 統合元の監査済み疑似ラベルデータと、その検証結果を保持します。
@dataclass(frozen=True, slots=True)
class _SourceDataset:
    manifest_path: Path
    root: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    sequence_id: str
    split: str
    video_sha256: str
    source_frames_total: int | None
    samples: tuple[dict[str, Any], ...]
    image_paths: Mapping[str, Path]


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


# レコード群をJSON Linesとして保存します。
def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            target.write("\n")


# 文字列が64桁の16進数SHA-256か判定します。
def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in text
    )


# 値が正しい形式のSHA-256であることを検証します。
def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    return str(value)


# 成果物パスをルート配下の相対ファイルとして検証します。
def _relative_file(root: Path, value: object) -> Path:
    root = root.resolve()
    path = (root / str(value)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"relative path escapes its dataset root: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


# 値を有限の数値へ変換し、NaN・無限大を拒否します。
def _finite_number(value: object, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


# 値が正の有限数であることを検証します。
def _positive_number(value: object, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


# 二つの数値が許容誤差内で等しいか判定します。
def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_GEOMETRY_REL_TOL,
        abs_tol=_GEOMETRY_ABS_TOL,
    )


# 入力データセットに必要なマニフェスト・画像・CSVがそろうか検証します。
def _validate_artifacts(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("Phase 7 manifest artifacts must be an object")
    verified: dict[str, Path] = {}
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping):
            raise TypeError(f"Phase 7 artifact {name!r} must be an object")
        if "relative_path" not in descriptor or "sha256" not in descriptor:
            raise ValueError(f"Phase 7 artifact {name!r} lacks path or SHA-256")
        path = _relative_file(root, descriptor["relative_path"])
        expected = _require_sha256(descriptor["sha256"], field=f"artifact {name} SHA-256")
        if sha256_file(path) != expected:
            raise ValueError(f"Phase 7 artifact SHA-256 mismatch: {name}")
        verified[str(name)] = path
    for required in ("samples_jsonl", "targets_csv", "split"):
        if required not in verified:
            raise ValueError(f"Phase 7 manifest lacks required artifact: {required}")
    return verified


# カメラ内部パラメーターと画像寸法の整合性を検証します。
def _validate_camera(camera: object, *, width: int, height: int) -> dict[str, Any]:
    if not isinstance(camera, Mapping):
        raise TypeError("sample camera must be an object")
    result = dict(camera)
    fx = _positive_number(result.get("fx_px"), field="camera.fx_px")
    fy = _positive_number(result.get("fy_px"), field="camera.fy_px")
    cx = _finite_number(result.get("cx_px"), field="camera.cx_px")
    cy = _finite_number(result.get("cy_px"), field="camera.cy_px")
    if not (-1.0 <= cx <= width) or not (-1.0 <= cy <= height):
        raise ValueError("sample camera principal point is implausibly outside the image")
    result.update({"fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy})
    return result


# ランドマーク番号・座標・信頼度の値を検証します。
def _validate_landmark(
    landmark: object,
    *,
    width: int,
    height: int,
    expected_index: int,
) -> dict[str, Any]:
    if not isinstance(landmark, Mapping):
        raise TypeError("feature landmark must be an object")
    result = dict(landmark)
    if int(result.get("landmark_index", -1)) != expected_index:
        raise ValueError("feature landmark order/index differs from source manifest")
    x = _finite_number(result.get("x_normalized"), field="landmark.x_normalized")
    y = _finite_number(result.get("y_normalized"), field="landmark.y_normalized")
    _finite_number(result.get("z_mediapipe_relative"), field="landmark.z_relative")
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise ValueError("accepted feature normalized coordinate is outside [0, 1]")
    u = int(result.get("u_px", -1))
    v = int(result.get("v_px", -1))
    if not 0 <= u < width or not 0 <= v < height:
        raise ValueError("accepted feature pixel coordinate is outside the image")
    if result.get("in_frame") is not True:
        raise ValueError("accepted feature landmark must be marked in_frame")
    if normalized_to_pixel(x, y, width=width, height=height) != (u, v):
        raise ValueError(
            "feature normalized coordinate does not reproduce its stored pixel "
            "with normalized_to_pixel"
        )
    return result


# 標本の識別子、入力画像、座標、教師深度、参照ハッシュを検証します。
def _validate_sample(
    sample: dict[str, Any],
    *,
    source_root: Path,
    sequence_id: str,
    split: str,
    feature_indices: Sequence[int],
    feature_names: Sequence[str],
    target_index: int,
    target_name: str,
) -> Path:
    if sample.get("sequence_id") != sequence_id or sample.get("split") != split:
        raise ValueError("Phase 7 sample sequence/split differs from its manifest")
    sample_id = str(sample.get("sample_id", ""))
    if not sample_id or ":aug=" in sample_id:
        raise ValueError("Phase 7 sample_id is empty or already augmented")
    if int(sample.get("frame_index", -1)) < 0 or int(sample.get("timestamp_ms", -1)) < 0:
        raise ValueError("Phase 7 frame index/timestamp must be non-negative")

    image = sample.get("image")
    if not isinstance(image, Mapping):
        raise TypeError("Phase 7 sample image must be an object")
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Phase 7 sample image dimensions must be positive")
    image_path = _relative_file(source_root, image.get("relative_path"))
    expected_png = _require_sha256(image.get("png_sha256"), field="image PNG SHA-256")
    expected_pixels = _require_sha256(
        image.get("bgr_pixel_sha256"), field="image BGR pixel SHA-256"
    )
    if sha256_file(image_path) != expected_png:
        raise ValueError(f"Phase 7 sample PNG SHA-256 mismatch: {sample_id}")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (height, width, 3):
        raise ValueError(f"Phase 7 sample image cannot be decoded at declared shape: {sample_id}")
    if pixel_sha256(bgr) != expected_pixels:
        raise ValueError(f"Phase 7 sample BGR pixel SHA-256 mismatch: {sample_id}")

    camera = _validate_camera(sample.get("camera"), width=width, height=height)
    hand = sample.get("hand")
    if not isinstance(hand, Mapping):
        raise TypeError("Phase 7 sample hand must be an object")
    _view_handedness(hand.get("handedness"), hflip=False)
    raw_features = hand.get("feature_landmarks")
    if not isinstance(raw_features, list) or len(raw_features) != len(feature_indices):
        raise ValueError("Phase 7 sample feature landmark count differs from manifest")
    for landmark, index, name in zip(
        raw_features,
        feature_indices,
        feature_names,
        strict=True,
    ):
        validated = _validate_landmark(
            landmark,
            width=width,
            height=height,
            expected_index=int(index),
        )
        if validated.get("landmark_name") != name:
            raise ValueError("feature landmark name differs from source manifest")

    target = sample.get("target")
    if not isinstance(target, Mapping):
        raise TypeError("Phase 7 sample target must be an object")
    if int(target.get("landmark_index", -1)) != target_index:
        raise ValueError("Phase 7 target index differs from manifest")
    if target.get("landmark_name") != target_name:
        raise ValueError("Phase 7 target name differs from manifest")
    if target.get("depth_sampling") != "single_pixel":
        raise ValueError("Phase 7 target must use single-pixel depth sampling")
    u = int(target.get("u_px", -1))
    v = int(target.get("v_px", -1))
    if not 0 <= u < width or not 0 <= v < height:
        raise ValueError("Phase 7 target pixel is outside the image")
    z = _positive_number(target.get("z_teacher_m"), field="target.z_teacher_m")
    xyz = target.get("camera_xyz_m")
    if not isinstance(xyz, Mapping):
        raise TypeError("Phase 7 target camera_xyz_m must be an object")
    x = _finite_number(xyz.get("x_m"), field="target.x_m")
    y = _finite_number(xyz.get("y_m"), field="target.y_m")
    xyz_z = _positive_number(xyz.get("z_m"), field="target.camera_z_m")
    expected_x = (u - camera["cx_px"]) * z / camera["fx_px"]
    expected_y = (v - camera["cy_px"]) * z / camera["fy_px"]
    if not (_close(x, expected_x) and _close(y, expected_y) and _close(xyz_z, z)):
        raise ValueError("Phase 7 target camera XYZ is inconsistent with pixel, K, and Z")
    if target.get("depth_valid") is not True:
        raise ValueError("Phase 7 accepted target must have depth_valid=true")

    teacher = sample.get("teacher")
    if not isinstance(teacher, Mapping):
        raise TypeError("Phase 7 sample teacher must be an object")
    if teacher.get("condition_id") != DEPTH_PRO_TEACHER_CONDITION_ID:
        raise ValueError("Phase 7 sample uses an unexpected teacher condition")
    if teacher.get("camera_mode") != "approx_focal" or teacher.get("depth_unit") != "metre":
        raise ValueError("Phase 7 sample teacher camera mode or depth unit is invalid")
    if not _close(
        _positive_number(teacher.get("input_focal_px"), field="teacher.input_focal_px"),
        camera["fx_px"],
    ):
        raise ValueError("Phase 7 sample teacher focal input differs from sample K")
    return image_path


# 教師値CSVの列・標本ID・数値が標本データと一致するか確認します。
def _validate_targets_csv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    if [str(row.get("sample_id", "")) for row in rows] != expected_ids:
        raise ValueError("Phase 7 targets.csv sample order differs from samples.jsonl")
    for row, sample in zip(rows, samples, strict=True):
        target = sample["target"]
        xyz = target["camera_xyz_m"]
        comparisons = {
            "u_px": float(target["u_px"]),
            "v_px": float(target["v_px"]),
            "x_m": float(xyz["x_m"]),
            "y_m": float(xyz["y_m"]),
            "z_teacher_m": float(target["z_teacher_m"]),
        }
        if any(not _close(float(row[key]), value) for key, value in comparisons.items()):
            raise ValueError("Phase 7 targets.csv values differ from samples.jsonl")
        exact = {
            "sequence_id": str(sample["sequence_id"]),
            "split": str(sample["split"]),
            "frame_index": str(sample["frame_index"]),
            "timestamp_ms": str(sample["timestamp_ms"]),
        }
        if any(str(row.get(key, "")) != value for key, value in exact.items()):
            raise ValueError("Phase 7 targets.csv metadata differs from samples.jsonl")


# 一つの入力疑似ラベルデータセットを読み込み、全参照を検証します。
def _load_source_dataset(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> _SourceDataset:
    manifest_path = path.resolve()
    root = manifest_path.parent
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        expected = _require_sha256(
            expected_manifest_sha256,
            field="expected source manifest SHA-256",
        )
        if manifest_sha256 != expected:
            raise ValueError(
                "source manifest SHA-256 mismatch before reading: expected "
                f"{expected}, observed {manifest_sha256}"
            )
    manifest = _read_json(manifest_path)
    if manifest.get("format") != PSEUDO_LABEL_DATASET_FORMAT:
        raise ValueError(f"unsupported Phase 7 dataset format: {manifest_path}")
    if manifest.get("format_version") != PSEUDO_LABEL_DATASET_FORMAT_VERSION:
        raise ValueError(f"unsupported Phase 7 dataset version: {manifest_path}")
    if manifest.get("ground_truth") is not False or manifest.get("label_type") != "pseudo_label":
        raise ValueError("Phase 7 source must identify its labels as pseudo-labels")

    sequence = manifest.get("sequence")
    if not isinstance(sequence, Mapping):
        raise TypeError("Phase 7 sequence metadata must be an object")
    sequence_id = str(sequence.get("id", ""))
    split = str(sequence.get("split", ""))
    if not sequence_id:
        raise ValueError("Phase 7 sequence ID must not be empty")
    if split not in _ALLOWED_SPLITS:
        raise ValueError("student aggregation accepts only train/validation sources")

    feature_meta = manifest.get("feature_landmarks")
    target_meta = manifest.get("target_landmark")
    if not isinstance(feature_meta, Mapping) or not isinstance(target_meta, Mapping):
        raise TypeError("Phase 7 landmark metadata is incomplete")
    feature_indices = tuple(int(index) for index in feature_meta.get("indices", []))
    if not feature_indices or len(set(feature_indices)) != len(feature_indices):
        raise ValueError("Phase 7 feature landmark indices must be non-empty and unique")
    feature_names = tuple(str(name) for name in feature_meta.get("names", []))
    if len(feature_names) != len(feature_indices) or len(set(feature_names)) != len(feature_names):
        raise ValueError("Phase 7 feature landmark names must be non-empty and unique")
    target_index = int(target_meta.get("index", -1))
    target_name = str(target_meta.get("name", ""))
    if not target_name or target_meta.get("depth_sampling") != "single_pixel":
        raise ValueError("Phase 7 target landmark metadata is invalid")

    teacher = manifest.get("teacher")
    if not isinstance(teacher, Mapping):
        raise TypeError("Phase 7 teacher metadata must be an object")
    if teacher.get("condition_id") != DEPTH_PRO_TEACHER_CONDITION_ID:
        raise ValueError("Phase 7 source uses an unexpected teacher condition")
    if teacher.get("camera_mode") != "approx_focal":
        raise ValueError("Phase 7 source teacher must use approx_focal mode")
    filtering = manifest.get("filtering")
    if filtering != _EXPECTED_FILTERING:
        raise ValueError(
            "Phase 7 source must use single-pixel extraction, no temporal filtering, "
            "and no label clipping"
        )
    camera_coordinate_system = manifest.get("camera_coordinate_system")
    if camera_coordinate_system != _EXPECTED_CAMERA_COORDINATE_SYSTEM:
        raise ValueError("Phase 7 source camera coordinate system is unsupported")

    artifacts = _validate_artifacts(root, manifest)
    samples = _read_jsonl(artifacts["samples_jsonl"])
    if not samples:
        raise ValueError(f"Phase 7 source contains no accepted samples: {sequence_id}")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or int(counts.get("accepted_samples", -1)) != len(samples):
        raise ValueError("Phase 7 accepted sample count differs from samples.jsonl")
    raw_source_frames_total = counts.get("source_frames_total")
    source_frames_total = None if raw_source_frames_total is None else int(raw_source_frames_total)
    if source_frames_total is not None:
        if source_frames_total <= 0:
            raise ValueError("Phase 7 source_frames_total must be positive")
        if any(int(sample.get("frame_index", -1)) >= source_frames_total for sample in samples):
            raise ValueError("Phase 7 sample frame index falls outside source_frames_total")
    sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Phase 7 source contains duplicate sample IDs")
    split_ids = artifacts["split"].read_text(encoding="utf-8").splitlines()
    if split_ids != sample_ids:
        raise ValueError("Phase 7 split artifact differs from samples.jsonl")
    _validate_targets_csv(artifacts["targets_csv"], samples)

    image_paths: dict[str, Path] = {}
    frame_keys: set[tuple[int, int]] = set()
    for sample in samples:
        image_path = _validate_sample(
            sample,
            source_root=root,
            sequence_id=sequence_id,
            split=split,
            feature_indices=feature_indices,
            feature_names=feature_names,
            target_index=target_index,
            target_name=target_name,
        )
        sample_id = str(sample["sample_id"])
        hand = sample["hand"]
        frame_key = (int(sample["frame_index"]), int(hand["hand_index"]))
        if frame_key in frame_keys:
            raise ValueError("Phase 7 source repeats a frame/hand sample")
        frame_keys.add(frame_key)
        image_paths[sample_id] = image_path

    prepared = manifest.get("prepared_inputs")
    if not isinstance(prepared, Mapping):
        raise TypeError("Phase 7 prepared_inputs metadata must be an object")
    source_meta = prepared.get("source")
    if not isinstance(source_meta, Mapping):
        raise TypeError("Phase 7 prepared source metadata must be an object")
    raw_prepared_frame_count = source_meta.get("source_frame_count")
    if raw_prepared_frame_count is not None:
        prepared_frame_count = int(raw_prepared_frame_count)
        if prepared_frame_count <= 0:
            raise ValueError("Phase 7 prepared source_frame_count must be positive")
        if source_frames_total is not None and prepared_frame_count != source_frames_total:
            raise ValueError("Phase 7 source frame counts are inconsistent")
    video_sha256 = _require_sha256(source_meta.get("video_sha256"), field="source video SHA-256")
    _require_sha256(prepared.get("manifest_sha256"), field="prepared manifest SHA-256")
    prepared_pixel_sha256 = _require_sha256(
        prepared.get("pixel_hash_sequence_sha256"),
        field="prepared pixel sequence SHA-256",
    )
    selection = teacher.get("selection_evidence")
    if selection is not None:
        if not isinstance(selection, Mapping):
            raise TypeError("Phase 7 teacher selection_evidence must be an object or null")
        if selection.get("condition_id") != DEPTH_PRO_TEACHER_CONDITION_ID:
            raise ValueError("teacher selection evidence condition differs from teacher")
        if selection.get("provenance_verified") is not True:
            raise ValueError("teacher selection evidence is not provenance-verified")
        _require_sha256(selection.get("report_sha256"), field="selection report SHA-256")
        if selection.get("source_video_sha256") != video_sha256:
            raise ValueError("teacher selection evidence source video differs from dataset")
        if selection.get("input_pixel_sequence_sha256") != prepared_pixel_sha256:
            raise ValueError("teacher selection evidence pixels differ from prepared inputs")
    model = teacher.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("Phase 7 teacher model metadata must be an object")
    if "checkpoint_sha256" in model:
        _require_sha256(model["checkpoint_sha256"], field="teacher checkpoint SHA-256")
    return _SourceDataset(
        manifest_path=manifest_path,
        root=root,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        sequence_id=sequence_id,
        split=split,
        video_sha256=video_sha256,
        source_frames_total=source_frames_total,
        samples=tuple(samples),
        image_paths=image_paths,
    )


# 教師モデルと生成設定の識別情報をまとめます。
def _teacher_fingerprint(manifest: Mapping[str, Any]) -> dict[str, Any]:
    teacher = manifest["teacher"]
    assert isinstance(teacher, Mapping)
    missing_teacher = [key for key in _STABLE_TEACHER_FIELDS if key not in teacher]
    if missing_teacher:
        raise ValueError(f"Phase 7 teacher metadata lacks stable fields: {missing_teacher}")
    model = teacher.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("Phase 7 teacher model metadata must be an object")
    missing_model = [key for key in _STABLE_MODEL_FIELDS if key not in model]
    if missing_model:
        raise ValueError(f"Phase 7 teacher model lacks stable fields: {missing_model}")
    _require_sha256(model["checkpoint_sha256"], field="teacher checkpoint SHA-256")
    for field in ("source_commit", "checkpoint_revision"):
        if not str(model[field]):
            raise ValueError(f"teacher model {field} must not be empty")
    return {
        **{key: copy.deepcopy(teacher[key]) for key in _STABLE_TEACHER_FIELDS},
        "model": {key: copy.deepcopy(model[key]) for key in _STABLE_MODEL_FIELDS},
    }


# 複数データセット間のID重複・分割・形式の整合性を検証します。
def _validate_sources(
    sources: Sequence[_SourceDataset],
    *,
    chronological_tail_split: bool,
) -> None:
    if len(sources) < 2:
        raise ValueError("student dataset requires multiple Phase 7 source datasets")
    sequence_ids = [source.sequence_id for source in sources]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("source sequence IDs must be globally unique")
    video_hashes = [source.video_sha256 for source in sources]
    if len(set(video_hashes)) != len(video_hashes):
        raise ValueError("source video SHA-256 values must be globally unique")
    if not chronological_tail_split:
        splits = {source.split for source in sources}
        if splits != _ALLOWED_SPLITS:
            raise ValueError(
                "student dataset requires at least one train and one validation sequence"
            )

    first = sources[0].manifest
    for source in sources[1:]:
        if source.manifest.get("feature_landmarks") != first.get("feature_landmarks"):
            raise ValueError("source datasets use different feature landmark definitions")
        if source.manifest.get("target_landmark") != first.get("target_landmark"):
            raise ValueError("source datasets use different target landmark definitions")
        if source.manifest.get("filtering") != first.get("filtering"):
            raise ValueError("source datasets use different filtering configurations")
        if source.manifest.get("camera_coordinate_system") != first.get("camera_coordinate_system"):
            raise ValueError("source datasets use different camera coordinate systems")
        if _teacher_fingerprint(source.manifest) != _teacher_fingerprint(first):
            raise ValueError("source datasets use different teacher configurations or checkpoints")

    if chronological_tail_split:
        pixel_sequences: dict[str, str] = {}
        for source in sources:
            for sample in source.samples:
                digest = str(sample["image"]["bgr_pixel_sha256"])
                previous = pixel_sequences.setdefault(digest, source.sequence_id)
                if previous != source.sequence_id:
                    raise ValueError(
                        "identity BGR pixel hashes overlap across source videos: "
                        f"{previous!r} and {source.sequence_id!r}"
                    )
    else:
        train_pixels = {
            str(sample["image"]["bgr_pixel_sha256"])
            for source in sources
            if source.split == "train"
            for sample in source.samples
        }
        validation_pixels = {
            str(sample["image"]["bgr_pixel_sha256"])
            for source in sources
            if source.split == "validation"
            for sample in source.samples
        }
        overlap = train_pixels & validation_pixels
        if overlap:
            raise ValueError(
                "identity BGR pixel hashes overlap across train and validation splits: "
                f"{len(overlap)} shared frame(s)"
            )


# 反転画像に対応する新しい標本IDと出典情報を作ります。
def _transfer_identity(source: Path, target: Path, *, mode: Literal["hardlink", "copy"]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


# 左右反転したRGB画像をPNGとして保存します。
def _write_flipped_png(path: Path, source_bgr: np.ndarray) -> np.ndarray:
    flipped = np.ascontiguousarray(source_bgr[:, ::-1, :])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), flipped, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"failed to write HFlip PNG: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or not np.array_equal(decoded, flipped):
        raise ValueError(f"HFlip PNG did not round-trip losslessly: {path}")
    return flipped


# 左右反転ビューに合わせて手の左右ラベルを入れ替えます。
def _view_handedness(value: object, *, hflip: bool) -> str | None:
    if value is None:
        return None
    handedness = str(value)
    if handedness not in {"Left", "Right"}:
        raise ValueError(f"unsupported MediaPipe handedness category: {handedness!r}")
    if not hflip:
        return handedness
    return "Left" if handedness == "Right" else "Right"


# 標本画像の元データ、ハッシュ、変換内容を記録します。
def _source_image_provenance(image: Mapping[str, Any]) -> dict[str, str]:
    return {
        "relative_path": str(image["relative_path"]),
        "png_sha256": str(image["png_sha256"]),
        "bgr_pixel_sha256": str(image["bgr_pixel_sha256"]),
    }


# 正規化されたX座標を左右反転し、Y座標は保ちます。
def _hflip_normalized_x(*, source_x: float, target_u: int, width: int) -> float:
    """Reflect MediaPipe x while preserving this project's exact pixel rounding.

    ``normalized_to_pixel`` uses ``round(x * width)`` with half-up rounding and
    clips the exact 1.0 boundary.  The subpixel reflection is attempted first;
    half-tie cases can land one pixel away, in which case ``target_u / width``
    is the deterministic pixel-consistent fallback.
    """

    candidate = min(
        max(((width - 1) - float(source_x) * width) / width, 0.0),
        1.0,
    )
    mapped_u, _ = normalized_to_pixel(candidate, 0.0, width=width, height=1)
    if mapped_u == target_u:
        return candidate
    fallback = target_u / width
    mapped_u, _ = normalized_to_pixel(fallback, 0.0, width=width, height=1)
    if mapped_u != target_u:
        raise AssertionError("could not construct a pixel-consistent HFlip normalized x")
    return fallback


# 元画像または左右反転画像と座標・教師値を一標本にまとめます。
def _make_view_sample(
    source_sample: Mapping[str, Any],
    *,
    variant: Literal["identity", "hflip"],
    split: Literal["train", "validation"] | None = None,
    output_image_path: Path,
    output_png_sha256: str,
    output_pixel_sha256: str,
) -> dict[str, Any]:
    sample = copy.deepcopy(dict(source_sample))
    base_id = str(source_sample["sample_id"])
    sequence_id = str(source_sample["sequence_id"])
    width = int(source_sample["image"]["width"])
    hflip = variant == "hflip"

    sample.update(
        {
            "schema_version": STUDENT_SAMPLE_SCHEMA_VERSION,
            "sample_id": f"{base_id}:aug={variant}",
            "source_sample_id": base_id,
            "source_sequence_id": sequence_id,
            "view_sequence_id": f"{sequence_id}@{variant}",
        }
    )
    if split is not None:
        sample["split"] = split
    sample["image"] = {
        **dict(source_sample["image"]),
        "relative_path": output_image_path.as_posix(),
        "png_sha256": output_png_sha256,
        "bgr_pixel_sha256": output_pixel_sha256,
    }

    source_camera = source_sample["camera"]
    camera = copy.deepcopy(dict(source_camera))
    if hflip:
        camera["cx_px"] = (width - 1) - float(source_camera["cx_px"])
        camera["coordinate_frame"] = "hflip_virtual_camera_x_right_y_down_z_forward"
        camera["derived_from_source_by"] = "x_axis_reflection"
    else:
        camera["coordinate_frame"] = "source_camera_x_right_y_down_z_forward"
        camera["derived_from_source_by"] = "identity"
    sample["camera"] = camera

    source_hand = source_sample["hand"]
    hand = copy.deepcopy(dict(source_hand))
    source_handedness = source_hand.get("handedness")
    view_handedness = _view_handedness(source_handedness, hflip=hflip)
    hand["handedness_detected_source"] = source_handedness
    hand["handedness_in_view"] = view_handedness
    hand["handedness"] = view_handedness
    if hflip:
        transformed_landmarks: list[dict[str, Any]] = []
        for raw in source_hand["feature_landmarks"]:
            landmark = copy.deepcopy(dict(raw))
            target_u = (width - 1) - int(raw["u_px"])
            landmark["x_normalized"] = _hflip_normalized_x(
                source_x=float(raw["x_normalized"]),
                target_u=target_u,
                width=width,
            )
            landmark["u_px"] = target_u
            transformed_landmarks.append(landmark)
        hand["feature_landmarks"] = transformed_landmarks
    sample["hand"] = hand

    source_target = source_sample["target"]
    target = copy.deepcopy(dict(source_target))
    if hflip:
        target["u_px"] = (width - 1) - int(source_target["u_px"])
        source_xyz = source_target["camera_xyz_m"]
        target["camera_xyz_m"] = {
            "x_m": -float(source_xyz["x_m"]),
            "y_m": float(source_xyz["y_m"]),
            "z_m": float(source_xyz["z_m"]),
        }
    sample["target"] = target

    teacher = copy.deepcopy(dict(source_sample["teacher"]))
    teacher["label_source_sample_id"] = base_id
    teacher["label_reused_from_source"] = hflip
    teacher["teacher_inference_on_augmented_image"] = False if hflip else None
    sample["teacher"] = teacher

    source_image = source_sample["image"]
    sample["augmentation"] = {
        "variant": variant,
        "materialized": True,
        "source_sample_id": base_id,
        "source_image": _source_image_provenance(source_image),
        "teacher_label_reused": hflip,
        "pixel_transform": "u_view = width - 1 - u_source; v_view = v_source"
        if hflip
        else "identity",
        "normalized_transform": (
            "discrete-pixel-center subpixel reflection with a pixel-consistent "
            "u_view/width fallback; y_view = y_source"
        )
        if hflip
        else "identity",
    }
    return sample


# 反転前後の画像・カメラ・座標・深度が幾何的に整合するか検証します。
def _verify_view_geometry(sample: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    variant = sample["augmentation"]["variant"]
    target = sample["target"]
    source_target = source["target"]
    if float(target["z_teacher_m"]) != float(source_target["z_teacher_m"]):
        raise ValueError("augmentation changed teacher Z")
    if variant == "hflip":
        width = int(sample["image"]["width"])
        if int(target["u_px"]) != width - 1 - int(source_target["u_px"]):
            raise ValueError("HFlip target pixel transform is inconsistent")
    width = int(sample["image"]["width"])
    height = int(sample["image"]["height"])
    for landmark in sample["hand"]["feature_landmarks"]:
        observed = normalized_to_pixel(
            float(landmark["x_normalized"]),
            float(landmark["y_normalized"]),
            width=width,
            height=height,
        )
        if observed != (int(landmark["u_px"]), int(landmark["v_px"])):
            raise ValueError("view landmark normalized/pixel coordinates are inconsistent")
    camera = sample["camera"]
    xyz = target["camera_xyz_m"]
    z = float(target["z_teacher_m"])
    expected_x = (int(target["u_px"]) - float(camera["cx_px"])) * z / float(camera["fx_px"])
    expected_y = (int(target["v_px"]) - float(camera["cy_px"])) * z / float(camera["fy_px"])
    if not (
        _close(float(xyz["x_m"]), expected_x)
        and _close(float(xyz["y_m"]), expected_y)
        and float(xyz["z_m"]) == z
    ):
        raise ValueError("augmented target XYZ is inconsistent with transformed pixel and K")


# 標本IDと教師深度を対応づけたCSVを保存します。
def _write_targets_csv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_sample_id",
        "source_sequence_id",
        "view_sequence_id",
        "split",
        "augmentation",
        "frame_index",
        "timestamp_ms",
        "u_px",
        "v_px",
        "x_m",
        "y_m",
        "z_teacher_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            target = sample["target"]
            xyz = target["camera_xyz_m"]
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "source_sample_id": sample["source_sample_id"],
                    "source_sequence_id": sample["source_sequence_id"],
                    "view_sequence_id": sample["view_sequence_id"],
                    "split": sample["split"],
                    "augmentation": sample["augmentation"]["variant"],
                    "frame_index": sample["frame_index"],
                    "timestamp_ms": sample["timestamp_ms"],
                    "u_px": target["u_px"],
                    "v_px": target["v_px"],
                    "x_m": xyz["x_m"],
                    "y_m": xyz["y_m"],
                    "z_teacher_m": target["z_teacher_m"],
                }
            )


# 関係するPythonソースのSHA-256を集めて再現性を記録します。
def _implementation_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parents[1]
    candidates = {
        "student_dataset.py": Path(__file__),
        "artifacts.py": package_dir / "artifacts.py",
        "coordinates.py": package_dir / "coordinates.py",
        "pseudo_labels.py": package_dir / "pseudo_labels.py",
        "video_cache.py": package_dir / "video_cache.py",
        "build_student_dataset.py": project_dir / "scripts" / "build_student_dataset.py",
        "pyproject.toml": project_dir / "pyproject.toml",
        "uv.lock": project_dir / "uv.lock",
    }
    return {name: sha256_file(path) for name, path in candidates.items() if path.is_file()}


# 複数のハッシュを順序付きで結合し、全体ダイジェストを作ります。
def _concat_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_require_sha256(value, field="ordered digest value").encode("ascii"))
    return digest.hexdigest()


# 検証済み疑似ラベル系列を統合し、訓練用反転ビューを含むデータセットを作成します。
def build_student_dataset(
    *,
    source_manifest_paths: Sequence[Path],
    output_dir: Path,
    frame_transfer_mode: Literal["hardlink", "copy"] = "hardlink",
    expected_source_manifest_sha256s: Sequence[str] | None = None,
    validation_tail_fraction: float | None = None,
) -> dict[str, Any]:
    """Combine Phase 7 datasets and materialize train-only horizontal flips."""

    if frame_transfer_mode not in {"hardlink", "copy"}:
        raise ValueError("frame_transfer_mode must be 'hardlink' or 'copy'")
    if not source_manifest_paths:
        raise ValueError("at least one source manifest is required")
    chronological_tail_split = validation_tail_fraction is not None
    if validation_tail_fraction is not None and (
        not math.isfinite(validation_tail_fraction) or not 0.0 < validation_tail_fraction < 1.0
    ):
        raise ValueError("validation_tail_fraction must be finite and in (0, 1)")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"student dataset output directory is not empty: {output_dir}")
    if expected_source_manifest_sha256s is not None and len(
        expected_source_manifest_sha256s
    ) != len(source_manifest_paths):
        raise ValueError(
            "expected_source_manifest_sha256s must contain one digest per source manifest"
        )

    expected_hashes: Sequence[str | None]
    if expected_source_manifest_sha256s is None:
        expected_hashes = [None] * len(source_manifest_paths)
    else:
        expected_hashes = list(expected_source_manifest_sha256s)
    sources = [
        _load_source_dataset(
            Path(path),
            expected_manifest_sha256=expected,
        )
        for path, expected in zip(source_manifest_paths, expected_hashes, strict=True)
    ]
    _validate_sources(
        sources,
        chronological_tail_split=chronological_tail_split,
    )
    sources.sort(key=lambda item: item.sequence_id)

    assigned_splits: dict[tuple[str, str], Literal["train", "validation"]] = {}
    chronological_boundaries: dict[str, dict[str, Any]] = {}
    for source in sources:
        if chronological_tail_split:
            if source.source_frames_total is None:
                raise ValueError(
                    "chronological tail splitting requires Phase 7 source_frames_total"
                )
            assert validation_tail_fraction is not None
            validation_frame_count = _ceil_fractional_count(
                source.source_frames_total,
                validation_tail_fraction,
            )
            validation_start_frame = source.source_frames_total - validation_frame_count
            train_count = 0
            validation_count = 0
            validation_indices: list[int] = []
            for source_sample in source.samples:
                frame_index = int(source_sample["frame_index"])
                assigned = "validation" if frame_index >= validation_start_frame else "train"
                assigned_splits[(source.sequence_id, str(source_sample["sample_id"]))] = assigned
                if assigned == "train":
                    train_count += 1
                else:
                    validation_count += 1
                    validation_indices.append(frame_index)
            if train_count == 0 or validation_count == 0:
                raise ValueError(
                    "chronological tail split must retain accepted train and validation "
                    f"samples for every source sequence: {source.sequence_id}"
                )
            chronological_boundaries[source.sequence_id] = {
                "source_frames_total": source.source_frames_total,
                "validation_frame_count": validation_frame_count,
                "validation_start_frame_inclusive": validation_start_frame,
                "train_frame_end_exclusive": validation_start_frame,
                "accepted_train_identity_samples": train_count,
                "accepted_validation_identity_samples": validation_count,
                "first_accepted_validation_frame": min(validation_indices),
                "last_accepted_validation_frame": max(validation_indices),
            }
        else:
            for source_sample in source.samples:
                assigned_splits[(source.sequence_id, str(source_sample["sample_id"]))] = (
                    source.split
                )  # type: ignore[assignment]

    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    output_images: dict[tuple[str, str, str], dict[str, Any]] = {}

    for source in sources:
        ordered_samples = sorted(
            source.samples,
            key=lambda row: (
                int(row["frame_index"]),
                int(row["hand"]["hand_index"]),
                str(row["sample_id"]),
            ),
        )
        for source_sample in ordered_samples:
            base_id = str(source_sample["sample_id"])
            assigned_split = assigned_splits[(source.sequence_id, base_id)]
            variants: tuple[Literal["identity", "hflip"], ...] = (
                ("identity", "hflip") if assigned_split == "train" else ("identity",)
            )
            source_image_path = source.image_paths[base_id]
            source_bgr = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
            if source_bgr is None:  # Already verified before output mutation.
                raise ValueError(f"source image became unreadable: {source_image_path}")
            source_name = source_image_path.name
            for variant in variants:
                image_key = (source.sequence_id, source_name, variant)
                relative_image = Path("frames") / source.sequence_id / variant / source_name
                target_image = output_dir / relative_image
                existing = output_images.get(image_key)
                if existing is None:
                    if variant == "identity":
                        _transfer_identity(
                            source_image_path,
                            target_image,
                            mode=frame_transfer_mode,
                        )
                        expected_bgr = source_bgr
                    else:
                        expected_bgr = _write_flipped_png(target_image, source_bgr)
                    decoded = cv2.imread(str(target_image), cv2.IMREAD_COLOR)
                    if decoded is None or not np.array_equal(decoded, expected_bgr):
                        raise ValueError(f"materialized image pixels differ: {target_image}")
                    existing = {
                        "relative_path": relative_image,
                        "png_sha256": sha256_file(target_image),
                        "bgr_pixel_sha256": pixel_sha256(decoded),
                    }
                    output_images[image_key] = existing
                view = _make_view_sample(
                    source_sample,
                    variant=variant,
                    split=assigned_split,
                    output_image_path=existing["relative_path"],
                    output_png_sha256=str(existing["png_sha256"]),
                    output_pixel_sha256=str(existing["bgr_pixel_sha256"]),
                )
                _verify_view_geometry(view, source_sample)
                samples.append(view)

    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("augmented sample IDs are not globally unique")
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    validation_samples = [sample for sample in samples if sample["split"] == "validation"]
    train_identity = [
        sample for sample in train_samples if sample["augmentation"]["variant"] == "identity"
    ]
    train_hflip = [
        sample for sample in train_samples if sample["augmentation"]["variant"] == "hflip"
    ]
    validation_hflip = [
        sample for sample in validation_samples if sample["augmentation"]["variant"] == "hflip"
    ]
    if len(train_identity) != len(train_hflip) or validation_hflip:
        raise AssertionError("train-doubling/validation-unaugmented invariant failed")
    train_source_ids = {str(sample["source_sample_id"]) for sample in train_samples}
    validation_source_ids = {str(sample["source_sample_id"]) for sample in validation_samples}
    if train_source_ids & validation_source_ids:
        raise AssertionError("source observation crosses train and validation splits")
    train_view_pixels = {str(sample["image"]["bgr_pixel_sha256"]) for sample in train_samples}
    validation_identity_pixels = {
        str(sample["image"]["bgr_pixel_sha256"]) for sample in validation_samples
    }
    pixel_overlap = train_view_pixels & validation_identity_pixels
    if pixel_overlap:
        raise ValueError(
            "identity BGR pixel hashes overlap across train and validation splits: "
            f"{len(pixel_overlap)} shared frame(s)"
        )

    samples_path = output_dir / "samples.jsonl"
    targets_path = output_dir / "targets.csv"
    train_split_path = output_dir / "splits" / "train.txt"
    validation_split_path = output_dir / "splits" / "validation.txt"
    _write_jsonl(samples_path, samples)
    _write_targets_csv(targets_path, samples)
    train_split_path.parent.mkdir(parents=True, exist_ok=True)
    train_split_path.write_text(
        "".join(f"{sample['sample_id']}\n" for sample in train_samples),
        encoding="utf-8",
    )
    validation_split_path.write_text(
        "".join(f"{sample['sample_id']}\n" for sample in validation_samples),
        encoding="utf-8",
    )

    selection_scopes: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    for source in sources:
        selection = source.manifest["teacher"].get("selection_evidence")
        if selection is not None:
            selection_scopes.append(
                {
                    "source_sequence_id": source.sequence_id,
                    "source_dataset_manifest_sha256": source.manifest_sha256,
                    "evidence": copy.deepcopy(selection),
                }
            )
        assigned_identity = [
            sample
            for sample in samples
            if sample["source_sequence_id"] == source.sequence_id
            and sample["augmentation"]["variant"] == "identity"
        ]
        assigned_split_names = sorted({sample["split"] for sample in assigned_identity})
        source_entries.append(
            {
                "sequence_id": source.sequence_id,
                "split": source.split,
                "source_manifest_split": source.split,
                "assigned_splits": assigned_split_names,
                "source_video_sha256": source.video_sha256,
                "dataset_manifest_path": str(source.manifest_path),
                "dataset_manifest_sha256": source.manifest_sha256,
                "prepared_manifest_sha256": source.manifest["prepared_inputs"].get(
                    "manifest_sha256"
                ),
                "source_frames_total": source.source_frames_total,
                "source_identity_samples": len(source.samples),
                "assigned_train_identity_samples": sum(
                    sample["split"] == "train" for sample in assigned_identity
                ),
                "assigned_validation_identity_samples": sum(
                    sample["split"] == "validation" for sample in assigned_identity
                ),
                "chronological_boundary": chronological_boundaries.get(source.sequence_id),
                "materialized_variants": (
                    ["identity", "hflip"] if "train" in assigned_split_names else ["identity"]
                ),
                "all_declared_artifact_hashes_verified": True,
                "all_sample_png_and_bgr_hashes_verified": True,
            }
        )

    png_hashes = [str(sample["image"]["png_sha256"]) for sample in samples]
    pixel_hashes = [str(sample["image"]["bgr_pixel_sha256"]) for sample in samples]
    first_manifest = sources[0].manifest
    teacher_fingerprint = _teacher_fingerprint(first_manifest)
    manifest: dict[str, Any] = {
        "format": STUDENT_DATASET_FORMAT,
        "format_version": STUDENT_DATASET_FORMAT_VERSION,
        "sample_schema_version": STUDENT_SAMPLE_SCHEMA_VERSION,
        "label_type": "pseudo_label",
        "ground_truth": False,
        "pseudo_label_notice": first_manifest.get("pseudo_label_notice"),
        "split_policy": (
            {
                "unit": "source video chronological frame tail",
                "random_frame_split": False,
                "temporal_order_preserved": True,
                "validation_tail_fraction": validation_tail_fraction,
                "fraction_denominator": (
                    "source_frames_total before hand-detection or teacher-label rejection"
                ),
                "validation_frame_count_formula": (
                    "ceil(source_frames_total * validation_tail_fraction)"
                ),
                "validation_membership": (
                    "frame_index >= source_frames_total - validation_frame_count"
                ),
                "split_assignment_uses_teacher_depth": False,
                "train_source_sequences": [source.sequence_id for source in sources],
                "validation_source_sequences": [source.sequence_id for source in sources],
                "per_sequence": chronological_boundaries,
                "augmented_views_inherit_source_split": False,
                "augmented_views_inherit_assigned_split": True,
                "validation_augmented": False,
            }
            if chronological_tail_split
            else {
                "unit": "source video sequence",
                "random_frame_split": False,
                "train_source_sequences": [
                    source.sequence_id for source in sources if source.split == "train"
                ],
                "validation_source_sequences": [
                    source.sequence_id for source in sources if source.split == "validation"
                ],
                "augmented_views_inherit_source_split": True,
                "validation_augmented": False,
            }
        ),
        "augmentation": {
            "strategy": "materialized_horizontal_reflection",
            "applies_to_splits": ["train"],
            "validation_policy": "identity-only",
            "image_formula": "I_h[v, width - 1 - u] = I[v, u]",
            "pixel_formula": "u_h = width - 1 - u; v_h = v",
            "normalized_landmark_formula": (
                "candidate x_h = clip(((width - 1) - x * width) / width); "
                "fallback x_h = u_h / width when half-up rounding misses u_h; y_h = y"
            ),
            "camera_intrinsics_formula": ("fx_h = fx; fy_h = fy; cx_h = width - 1 - cx; cy_h = cy"),
            "camera_xyz_formula": "X_h = -X; Y_h = Y; Z_h = Z",
            "teacher_depth_invariant": True,
            "teacher_inference_on_hflip": False,
            "landmark_indices_invariant": True,
            "mediapipe_relative_z_invariant": True,
            "handedness_policy": (
                "preserve handedness_detected_source; swap Left/Right in handedness_in_view"
            ),
            "pixel_coordinate_authority": (
                "integer pixels use exact width-1-u reflection; normalized x preserves the "
                "subpixel reflection when possible and is verified to round back to that pixel"
            ),
        },
        "feature_landmarks": copy.deepcopy(first_manifest["feature_landmarks"]),
        "target_landmark": copy.deepcopy(first_manifest["target_landmark"]),
        "teacher": {
            **teacher_fingerprint,
            "selection_evidence_scopes": selection_scopes,
            "hflip_labels_reused_from_identity": True,
        },
        "camera_coordinate_system": {
            "identity": "source camera: x-right, y-down, z-forward",
            "hflip": ("virtual reflected camera: x-right in the flipped image, y-down, z-forward"),
            "trajectory_policy": (
                "source Phase 7 trajectories remain identity-only and are not merged as "
                "augmented physical trajectories"
            ),
        },
        "sources": source_entries,
        "counts": {
            "source_sequences_total": len(sources),
            "train_source_sequences": len(
                {sample["source_sequence_id"] for sample in train_identity}
            ),
            "validation_source_sequences": len(
                {sample["source_sequence_id"] for sample in validation_samples}
            ),
            "source_identity_samples_total": sum(len(source.samples) for source in sources),
            "train_identity_samples": len(train_identity),
            "train_hflip_samples": len(train_hflip),
            "train_samples_total": len(train_samples),
            "train_exactly_doubled": len(train_samples) == 2 * len(train_identity),
            "validation_identity_samples": len(validation_samples),
            "validation_hflip_samples": len(validation_hflip),
            "validation_samples_total": len(validation_samples),
            "validation_unaugmented": len(validation_hflip) == 0,
            "samples_total": len(samples),
            "materialized_image_files": len(output_images),
        },
        "artifacts": {
            "samples_jsonl": {
                "relative_path": "samples.jsonl",
                "sha256": sha256_file(samples_path),
            },
            "targets_csv": {
                "relative_path": "targets.csv",
                "sha256": sha256_file(targets_path),
            },
            "train_split": {
                "relative_path": "splits/train.txt",
                "sha256": sha256_file(train_split_path),
            },
            "validation_split": {
                "relative_path": "splits/validation.txt",
                "sha256": sha256_file(validation_split_path),
            },
            "ordered_output_png_hashes_sha256": _concat_digest(png_hashes),
            "ordered_output_bgr_pixel_hashes_sha256": pixel_hash_sequence_sha256(pixel_hashes),
            "ordered_hash_encoding": (
                "SHA-256 of concatenated lowercase hexadecimal per-sample SHA-256 digests "
                "encoded as ASCII, in samples.jsonl order"
            ),
        },
        "provenance": {
            "source_manifests_fully_validated_before_output": True,
            "source_manifest_sha256_pins_supplied": (expected_source_manifest_sha256s is not None),
            "split_assignment_precedes_augmentation": True,
            "implementation_sha256": _implementation_hashes(),
            "frame_transfer_mode_for_identity": frame_transfer_mode,
            "hflip_materialization": "lossless PNG compression level 3",
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        },
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


__all__ = [
    "STUDENT_DATASET_FORMAT",
    "STUDENT_DATASET_FORMAT_VERSION",
    "STUDENT_SAMPLE_SCHEMA_VERSION",
    "build_student_dataset",
]
