'学習済みモデルを動画に適用し、指先の深度・三次元軌跡と画像上の注釈を描画するデモを生成します。'

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import write_json
from .camera import CameraIntrinsics
from .geometry import CAMERA_COORDINATE_CONVENTION, backproject_pixel
from .student_model import FingertipDepthStudent, StudentModelConfig
from .student_training import (
    STUDENT_CHECKPOINT_FORMAT_VERSION,
    TRAINING_RUN_FORMAT,
    TRAINING_RUN_FORMAT_VERSION,
    FingertipStudentDataset,
    StudentTrainingConfig,
    StudentTrainingSample,
    load_student_corpus,
)
from .trajectory import TrajectoryPoint, contiguous_segments, write_trajectory_csv
from .video_cache import sha256_file

DEMO_FORMAT = "fingertip-depth-student-trajectory-demo"
DEMO_FORMAT_VERSION = 1
CHECKPOINT_FORMAT = "fingertip-depth-student-checkpoint"
FRAME_CACHE_FORMAT = "fingertip-depth-lossless-video-frame-cache"
FRAME_CACHE_FORMAT_VERSION = 1
DEMO_SEQUENCE_ID = "finger_movement_2030"


# デモ動画のフレーム、手指の観測、画像情報を保持します。
@dataclass(frozen=True, slots=True)
class DemoObservation:
    """One identity-view sample and the pixels used for its overlay."""

    sample: StudentTrainingSample
    timestamp_ms: int
    landmark_pixels: tuple[tuple[int, int], ...]

    # 観測済みの指先位置を画像上のピクセル座標として返します。
    @property
    def fingertip_pixel(self) -> tuple[int, int]:
        return self.landmark_pixels[-1]


# デモの一フレームについて、推定深度と指先位置を保持します。
@dataclass(frozen=True, slots=True)
class PredictedObservation:
    """One student prediction with its camera-coordinate point."""

    observation: DemoObservation
    trajectory_point: TrajectoryPoint


# 動画からデコード済みの画像とフレーム時刻を保持します。
@dataclass(frozen=True, slots=True)
class CachedFrame:
    """One lossless source-video frame."""

    frame_index: int
    timestamp_ms: int
    path: Path
    width: int
    height: int


# 軌跡デモに必要な動画、モデル、設定、フレームキャッシュをまとめます。
@dataclass(frozen=True, slots=True)
class DemoInputs:
    """Validated inputs needed by inference and rendering."""

    run_manifest_path: Path
    run_manifest_sha256: str
    run_manifest: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    source_dataset_manifest_path: Path
    source_dataset_manifest_sha256: str
    source_video_path: Path
    source_video_sha256: str
    frame_cache_manifest_path: Path
    frame_cache_manifest_sha256: str
    frames: tuple[CachedFrame, ...]
    fps: float
    validation_start_frame: int
    intrinsics: CameraIntrinsics
    observations: tuple[DemoObservation, ...]
    selected_sample_ids: frozenset[str]


# JSONファイルを読み込みます。
def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


# 値が辞書形式であることを検証します。
def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


# 値が正しい形式のSHA-256であることを検証します。
def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    return digest


# 成果物ファイルの存在と期待されるSHA-256を検証します。
def _verified_file(path: Path, expected_sha256: object, *, field: str) -> tuple[Path, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _require_sha256(expected_sha256, field=f"{field} SHA-256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{field} SHA-256 mismatch: expected {expected}, observed {observed}")
    return path, observed


# 解決後のパスが許可された基準ディレクトリ内か確認します。
def _contained_path(root: Path, relative_path: object, *, field: str) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} escapes its manifest directory")
    return path


# マニフェスト上の成果物名を検証済み実ファイルパスに解決します。
def _artifact_path(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> tuple[Path, str]:
    artifacts = _require_mapping(manifest.get("artifacts"), field="artifacts")
    entry = _require_mapping(artifacts.get(name), field=f"artifacts.{name}")
    path = _contained_path(
        manifest_path.parent,
        entry.get("relative_path"),
        field=f"artifacts.{name}.relative_path",
    )
    return _verified_file(path, entry.get("sha256"), field=f"artifact {name}")


# デモ対象の標本ID一覧をマニフェストから読み込みます。
def _load_selected_ids(run_manifest_path: Path, run_manifest: Mapping[str, Any]) -> frozenset[str]:
    selected: set[str] = set()
    for name in ("included_train", "included_validation"):
        path, _digest = _artifact_path(run_manifest_path, run_manifest, name)
        for sample_id in path.read_text(encoding="utf-8").splitlines():
            if sample_id:
                selected.add(sample_id)
    if not selected:
        raise ValueError("training run selected no samples")
    return frozenset(selected)


# 画像に重ねるランドマーク・表示条件を読み込みます。
def _load_overlay_metadata(
    samples_path: Path,
    *,
    samples: Sequence[StudentTrainingSample],
) -> tuple[DemoObservation, ...]:
    wanted = {sample.sample_id: sample for sample in samples}
    metadata: dict[str, DemoObservation] = {}
    with samples_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TypeError(f"samples JSONL row {line_number} is not an object")
            sample_id = str(row.get("sample_id", ""))
            sample = wanted.get(sample_id)
            if sample is None:
                continue
            timestamp_ms = int(row.get("timestamp_ms", -1))
            if timestamp_ms < 0:
                raise ValueError(f"sample timestamp is invalid: {sample_id}")
            hand = _require_mapping(row.get("hand"), field=f"sample {sample_id} hand")
            raw_landmarks = hand.get("feature_landmarks")
            if not isinstance(raw_landmarks, list):
                raise TypeError(f"sample {sample_id} feature_landmarks must be an array")
            pixels_by_index: dict[int, tuple[int, int]] = {}
            for raw in raw_landmarks:
                landmark = _require_mapping(raw, field=f"sample {sample_id} landmark")
                index = int(landmark.get("landmark_index", -1))
                pixels_by_index[index] = (
                    int(landmark.get("u_px", -1)),
                    int(landmark.get("v_px", -1)),
                )
            try:
                landmark_pixels = tuple(
                    pixels_by_index[index] for index in sample_landmark_indices(sample)
                )
            except KeyError as error:
                raise ValueError(f"sample {sample_id} is missing a selected landmark") from error
            if any(
                not (0 <= u_px < sample.width and 0 <= v_px < sample.height)
                for u_px, v_px in landmark_pixels
            ):
                raise ValueError(f"sample {sample_id} contains an out-of-frame landmark")
            metadata[sample_id] = DemoObservation(
                sample=sample,
                timestamp_ms=timestamp_ms,
                landmark_pixels=landmark_pixels,
            )
    missing = set(wanted) - set(metadata)
    if missing:
        raise ValueError(f"missing overlay metadata for {len(missing)} selected samples")
    observations = tuple(sorted(metadata.values(), key=lambda value: value.sample.frame_index))
    frame_indices = [observation.sample.frame_index for observation in observations]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("demo source contains multiple selected observations for one frame")
    return observations


# デモ入力で採用されているランドマーク番号を返します。
def sample_landmark_indices(sample: StudentTrainingSample) -> tuple[int, ...]:
    """Return the fixed Phase 8 landmark order for a loaded training sample."""

    # StudentTrainingSampleはランドマーク名ではなく座標だけを保持します。
    # Phase 8のコーパス検証で、選択設定に対する座標の順序が保証されています。
    count = len(sample.landmark_xy)
    if count != 4:
        raise ValueError(f"trajectory demo expects four index-finger landmarks, got {count}")
    return (5, 6, 7, 8)


# 監査済み動画キャッシュのフレームと時刻一覧を読み込みます。
def _load_cached_frames(
    manifest_path: Path,
    *,
    expected_sha256: object,
    expected_video_path: Path,
    expected_video_sha256: str,
    expected_frame_count: int,
) -> tuple[tuple[CachedFrame, ...], float, str]:
    manifest_path, digest = _verified_file(
        manifest_path,
        expected_sha256,
        field="frame cache manifest",
    )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("format") != FRAME_CACHE_FORMAT
        or manifest.get("format_version") != FRAME_CACHE_FORMAT_VERSION
    ):
        raise ValueError("unsupported frame cache manifest")
    if Path(str(manifest.get("source", ""))).resolve() != expected_video_path.resolve():
        raise ValueError("frame cache source video differs from the teacher dataset")
    if manifest.get("source_sha256") != expected_video_sha256:
        raise ValueError("frame cache source video SHA-256 differs from the teacher dataset")
    fps = float(manifest.get("fps", math.nan))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("frame cache FPS must be finite and positive")
    if int(manifest.get("frame_count", -1)) != expected_frame_count:
        raise ValueError("frame cache frame count differs from the student dataset")
    rows = manifest.get("frames")
    if not isinstance(rows, list) or len(rows) != expected_frame_count:
        raise ValueError("frame cache rows differ from its declared frame count")
    frames: list[CachedFrame] = []
    previous_timestamp = -1
    for expected_index, raw in enumerate(rows):
        row = _require_mapping(raw, field=f"frame cache row {expected_index}")
        frame_index = int(row.get("frame_index", -1))
        timestamp_ms = int(row.get("timestamp_ms", -1))
        width = int(row.get("width", 0))
        height = int(row.get("height", 0))
        if frame_index != expected_index:
            raise ValueError("frame cache indices must be contiguous from zero")
        if timestamp_ms < 0 or timestamp_ms <= previous_timestamp and frame_index > 0:
            raise ValueError("frame cache timestamps must be non-negative and increasing")
        if width <= 0 or height <= 0:
            raise ValueError("frame cache dimensions must be positive")
        path = _contained_path(
            manifest_path.parent,
            row.get("relative_path"),
            field=f"frame cache row {frame_index} path",
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(CachedFrame(frame_index, timestamp_ms, path, width, height))
        previous_timestamp = timestamp_ms
    if len({(frame.width, frame.height) for frame in frames}) != 1:
        raise ValueError("trajectory demo requires a constant source frame size")
    return tuple(frames), fps, digest


# デモの設定・モデル・フレームキャッシュを読み込み、参照先を検証します。
def load_demo_inputs(run_manifest_path: Path) -> DemoInputs:
    """Resolve and validate the fixed ``finger_movement_2030`` demo inputs."""

    run_manifest_path = run_manifest_path.resolve()
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    run_manifest_sha256 = sha256_file(run_manifest_path)
    run_manifest = _read_json(run_manifest_path)
    if (
        run_manifest.get("format") != TRAINING_RUN_FORMAT
        or run_manifest.get("format_version") != TRAINING_RUN_FORMAT_VERSION
    ):
        raise ValueError("unsupported student training run manifest")

    checkpoint_path, checkpoint_sha256 = _artifact_path(
        run_manifest_path,
        run_manifest,
        "best_checkpoint",
    )
    dataset_entry = _require_mapping(run_manifest.get("dataset"), field="dataset")
    dataset_manifest_path, dataset_manifest_sha256 = _verified_file(
        Path(str(dataset_entry.get("manifest_path", ""))),
        dataset_entry.get("manifest_sha256"),
        field="student dataset manifest",
    )
    model_entry = _require_mapping(run_manifest.get("model"), field="model")
    model_config = _require_mapping(model_entry.get("config"), field="model.config")
    landmark_indices = tuple(int(value) for value in model_config.get("landmark_indices", []))
    if landmark_indices != (5, 6, 7, 8):
        raise ValueError("trajectory demo requires Phase 8 landmarks 5, 6, 7, and 8")
    corpus = load_student_corpus(
        dataset_manifest_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        landmark_indices=landmark_indices,
    )
    selected_sample_ids = _load_selected_ids(run_manifest_path, run_manifest)
    identity_samples = tuple(
        sorted(
            (
                sample
                for sample in (*corpus.train_samples, *corpus.validation_samples)
                if sample.source_sequence_id == DEMO_SEQUENCE_ID
                and sample.augmentation_variant == "identity"
                and sample.sample_id in selected_sample_ids
            ),
            key=lambda sample: sample.frame_index,
        )
    )
    if not identity_samples:
        raise ValueError(f"training run has no selected identity samples for {DEMO_SEQUENCE_ID}")

    artifacts = _require_mapping(corpus.manifest.get("artifacts"), field="dataset artifacts")
    samples_entry = _require_mapping(
        artifacts.get("samples_jsonl"), field="dataset artifacts.samples_jsonl"
    )
    samples_path = _contained_path(
        dataset_manifest_path.parent,
        samples_entry.get("relative_path"),
        field="dataset samples_jsonl path",
    )
    _verified_file(
        samples_path,
        samples_entry.get("sha256"),
        field="student samples JSONL",
    )
    observations = _load_overlay_metadata(samples_path, samples=identity_samples)

    source_rows = corpus.manifest.get("sources")
    if not isinstance(source_rows, list):
        raise TypeError("student dataset sources must be an array")
    matching_sources = [
        _require_mapping(row, field="student dataset source")
        for row in source_rows
        if isinstance(row, Mapping) and row.get("sequence_id") == DEMO_SEQUENCE_ID
    ]
    if len(matching_sources) != 1:
        raise ValueError(f"expected exactly one source entry for {DEMO_SEQUENCE_ID}")
    source = matching_sources[0]
    source_frame_count = int(source.get("source_frames_total", -1))
    source_dataset_manifest_path, source_dataset_manifest_sha256 = _verified_file(
        Path(str(source.get("dataset_manifest_path", ""))),
        source.get("dataset_manifest_sha256"),
        field="Depth Pro source dataset manifest",
    )
    source_dataset = _read_json(source_dataset_manifest_path)
    prepared_inputs = _require_mapping(
        source_dataset.get("prepared_inputs"), field="Depth Pro prepared_inputs"
    )
    prepared_source = _require_mapping(
        prepared_inputs.get("source"), field="Depth Pro prepared_inputs.source"
    )
    source_video_path, source_video_sha256 = _verified_file(
        Path(str(prepared_source.get("video_path", ""))),
        prepared_source.get("video_sha256"),
        field="source video",
    )
    if source_video_sha256 != source.get("source_video_sha256"):
        raise ValueError("student and teacher manifests disagree on source video SHA-256")
    frame_cache_manifest_path = Path(str(prepared_source.get("frame_cache_manifest_path", "")))
    frames, fps, frame_cache_manifest_sha256 = _load_cached_frames(
        frame_cache_manifest_path,
        expected_sha256=prepared_source.get("frame_cache_manifest_sha256"),
        expected_video_path=source_video_path,
        expected_video_sha256=source_video_sha256,
        expected_frame_count=source_frame_count,
    )

    camera = _require_mapping(source_dataset.get("camera"), field="Depth Pro camera")
    intrinsics_raw = _require_mapping(camera.get("intrinsics"), field="Depth Pro camera.intrinsics")
    intrinsics = CameraIntrinsics(
        fx_px=float(intrinsics_raw.get("fx_px")),
        fy_px=float(intrinsics_raw.get("fy_px")),
        cx_px=float(intrinsics_raw.get("cx_px")),
        cy_px=float(intrinsics_raw.get("cy_px")),
    )
    expected_size = _require_mapping(camera.get("image_size_px"), field="camera.image_size_px")
    if (frames[0].width, frames[0].height) != (
        int(expected_size.get("width", -1)),
        int(expected_size.get("height", -1)),
    ):
        raise ValueError("camera intrinsics image size differs from cached frames")
    for observation in observations:
        frame = frames[observation.sample.frame_index]
        if observation.timestamp_ms != frame.timestamp_ms:
            raise ValueError("student sample timestamp differs from the source frame cache")
        if (observation.sample.width, observation.sample.height) != (frame.width, frame.height):
            raise ValueError("student sample dimensions differ from the source frame cache")

    split_policy = _require_mapping(corpus.manifest.get("split_policy"), field="split_policy")
    per_sequence = _require_mapping(
        split_policy.get("per_sequence"), field="split_policy.per_sequence"
    )
    sequence_split = _require_mapping(
        per_sequence.get(DEMO_SEQUENCE_ID),
        field=f"split_policy.per_sequence.{DEMO_SEQUENCE_ID}",
    )
    validation_start_frame = int(sequence_split.get("validation_start_frame_inclusive", -1))
    if not 0 < validation_start_frame < len(frames):
        raise ValueError("validation split boundary is outside the source video")

    return DemoInputs(
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha256,
        run_manifest=run_manifest,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
        source_dataset_manifest_path=source_dataset_manifest_path,
        source_dataset_manifest_sha256=source_dataset_manifest_sha256,
        source_video_path=source_video_path,
        source_video_sha256=source_video_sha256,
        frame_cache_manifest_path=frame_cache_manifest_path.resolve(),
        frame_cache_manifest_sha256=frame_cache_manifest_sha256,
        frames=frames,
        fps=fps,
        validation_start_frame=validation_start_frame,
        intrinsics=intrinsics,
        observations=observations,
        selected_sample_ids=selected_sample_ids,
    )


# チェックポイント内の値からモデル構成を復元します。
def _model_config_from_checkpoint(raw: object) -> StudentModelConfig:
    config = _require_mapping(raw, field="checkpoint model_config")
    allowed = {field.name for field in fields(StudentModelConfig)}
    values = {name: config[name] for name in allowed if name in config}
    values["pretrained_image_encoder"] = False
    if "landmark_indices" in values:
        values["landmark_indices"] = tuple(int(value) for value in values["landmark_indices"])
    return StudentModelConfig(**values)


# チェックポイント内の値から学習時の構成を復元します。
def _training_config_from_checkpoint(raw: object) -> StudentTrainingConfig:
    config = _require_mapping(raw, field="checkpoint training_config")
    allowed = {field.name for field in fields(StudentTrainingConfig)}
    values = {name: config[name] for name in allowed if name in config}
    if "image_mean" in values:
        values["image_mean"] = tuple(float(value) for value in values["image_mean"])
    if "image_std" in values:
        values["image_std"] = tuple(float(value) for value in values["image_std"])
    return StudentTrainingConfig(**values)


# 指定条件と利用可能な演算装置から推論デバイスを選びます。
def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device_name}")
    return device


# チェックポイントと設定から生徒モデルを復元し、推論状態にします。
def load_student_model(
    inputs: DemoInputs,
    *,
    device_name: str,
) -> tuple[FingertipDepthStudent, StudentTrainingConfig, torch.device, Mapping[str, Any]]:
    """Load the trained student without downloading encoder weights."""

    device = _resolve_device(device_name)
    checkpoint = torch.load(inputs.checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("student checkpoint must be a mapping")
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("format_version") != STUDENT_CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError("unsupported student checkpoint")
    if checkpoint.get("dataset_manifest_sha256") != inputs.dataset_manifest_sha256:
        raise ValueError("checkpoint and run manifest refer to different student datasets")
    model_config = _model_config_from_checkpoint(checkpoint.get("model_config"))
    training_config = _training_config_from_checkpoint(checkpoint.get("training_config"))
    model = FingertipDepthStudent(model_config, initial_depth_bias_m=0.0)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, training_config, device, checkpoint


# 動画フレームを生徒モデルに通し、指先深度の予測時系列を作成します。
def predict_trajectory(
    inputs: DemoInputs,
    *,
    model: FingertipDepthStudent,
    training_config: StudentTrainingConfig,
    device: torch.device,
    batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[PredictedObservation, ...]:
    """Run the student on the exact identity-view images used by Phase 8."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    samples = tuple(observation.sample for observation in inputs.observations)
    dataset = FingertipStudentDataset(
        samples,
        image_size=training_config.image_size,
        image_mean=training_config.image_mean,
        image_std=training_config.image_std,
        preload_images=False,
        verify_png_sha256=training_config.verify_image_png_sha256,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[float] = []
    use_bfloat16 = training_config.precision == "bfloat16" and device.type == "cuda"
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["pixel_values"].to(device=device, non_blocking=True)
            landmarks = batch["landmark_coordinates"].to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                values = model(images, landmarks)
            predictions.extend(float(value) for value in values.float().cpu().tolist())
            if progress is not None:
                progress(f"inference batch {batch_index}/{len(loader)}")
    if len(predictions) != len(inputs.observations):
        raise AssertionError("prediction count differs from the selected observations")

    result: list[PredictedObservation] = []
    for observation, prediction_m in zip(inputs.observations, predictions, strict=True):
        if not math.isfinite(prediction_m) or prediction_m <= 0.0:
            raise ValueError(
                f"student prediction must be positive and finite: "
                f"frame {observation.sample.frame_index}, z={prediction_m}"
            )
        u_px, v_px = observation.fingertip_pixel
        camera_point = backproject_pixel(
            u_px=float(u_px),
            v_px=float(v_px),
            z_m=prediction_m,
            intrinsics=inputs.intrinsics,
        )
        point = TrajectoryPoint.from_camera_point(
            frame_index=observation.sample.frame_index,
            timestamp_ms=observation.timestamp_ms,
            u_px=float(u_px),
            v_px=float(v_px),
            point=camera_point,
        )
        result.append(PredictedObservation(observation=observation, trajectory_point=point))
    return tuple(result)


# 描画対象の座標範囲に余白を足して表示範囲を決めます。
def _padded_bounds(values: Sequence[float], *, minimum_span: float) -> tuple[float, float]:
    if not values:
        raise ValueError("plot bounds require at least one value")
    low = float(min(values))
    high = float(max(values))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("plot bounds must be finite")
    span = max(high - low, minimum_span)
    centre = (low + high) / 2.0
    padding = span * 0.08
    return centre - span / 2.0 - padding, centre + span / 2.0 + padding


# 文字に影を付けて背景とのコントラストを高めます。
def _put_text_with_shadow(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# グラフのデータ座標を画像上の座標へ変換する処理を返します。
def _plot_transform(
    *,
    x_m: float,
    z_m: float,
    x_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[int, int]:
    x_fraction = (x_m - x_bounds[0]) / (x_bounds[1] - x_bounds[0])
    z_fraction = (z_m - z_bounds[0]) / (z_bounds[1] - z_bounds[0])
    return (
        round(left + x_fraction * (right - left)),
        round(bottom - z_fraction * (bottom - top)),
    )


# カメラの前後・上下方向から見た指先軌跡を描画します。
def _draw_xz_panel(
    *,
    height: int,
    width: int,
    visible_points: Sequence[TrajectoryPoint],
    all_points: Sequence[TrajectoryPoint],
    frame_index: int,
    validation_start_frame: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), (27, 30, 36), dtype=np.uint8)
    left, right = 105, width - 48
    top, bottom = 170, height - 145
    if right <= left or bottom <= top:
        raise ValueError("plot panel is too small")
    x_bounds = _padded_bounds([point.x_m for point in all_points], minimum_span=0.02)
    z_bounds = _padded_bounds([point.z_m for point in all_points], minimum_span=0.05)

    cv2.rectangle(panel, (left, top), (right, bottom), (90, 96, 108), 2)
    for tick in range(6):
        fraction = tick / 5.0
        x_px = round(left + fraction * (right - left))
        y_px = round(bottom - fraction * (bottom - top))
        cv2.line(panel, (x_px, top), (x_px, bottom), (52, 57, 66), 1, cv2.LINE_AA)
        cv2.line(panel, (left, y_px), (right, y_px), (52, 57, 66), 1, cv2.LINE_AA)
        x_cm = 100.0 * (x_bounds[0] + fraction * (x_bounds[1] - x_bounds[0]))
        z_cm = 100.0 * (z_bounds[0] + fraction * (z_bounds[1] - z_bounds[0]))
        cv2.putText(
            panel,
            f"{x_cm:.1f}",
            (x_px - 24, bottom + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (185, 190, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{z_cm:.1f}",
            (8, y_px + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (185, 190, 200),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        "Predicted fingertip trajectory",
        (42, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "X right / Z forward (camera coordinates)",
        (42, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 187, 198),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "X [cm]",
        ((left + right) // 2 - 30, height - 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 215, 225),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Z [cm]",
        (12, top - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 215, 225),
        2,
        cv2.LINE_AA,
    )

    if visible_points:
        for segment in contiguous_segments(visible_points):
            pixels = np.asarray(
                [
                    _plot_transform(
                        x_m=point.x_m,
                        z_m=point.z_m,
                        x_bounds=x_bounds,
                        z_bounds=z_bounds,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                    )
                    for point in segment
                ],
                dtype=np.int32,
            )
            if len(pixels) >= 2:
                cv2.polylines(panel, [pixels], False, (255, 170, 55), 4, cv2.LINE_AA)
        current = visible_points[-1]
        current_pixel = _plot_transform(
            x_m=current.x_m,
            z_m=current.z_m,
            x_bounds=x_bounds,
            z_bounds=z_bounds,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )
        cv2.circle(panel, current_pixel, 11, (35, 65, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, current_pixel, 13, (245, 245, 245), 2, cv2.LINE_AA)
        depth_text = f"Current Z: {current.z_m * 100.0:.1f} cm"
    else:
        depth_text = "Current Z: --"
    cv2.putText(
        panel,
        depth_text,
        (42, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )

    is_validation = frame_index >= validation_start_frame
    phase_text = "VALID tail (same video)" if is_validation else "TRAIN segment"
    phase_color = (80, 180, 255) if is_validation else (90, 220, 130)
    cv2.rectangle(panel, (35, height - 58), (width - 35, height - 20), (44, 48, 56), -1)
    cv2.putText(
        panel,
        phase_text,
        (50, height - 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        phase_color,
        2,
        cv2.LINE_AA,
    )
    return panel


# 一フレームの画像、予測値、軌跡パネルを合成して描画します。
def render_demo_frame(
    source_bgr: np.ndarray,
    *,
    frame_index: int,
    current_observation: PredictedObservation | None,
    visible_points: Sequence[TrajectoryPoint],
    all_points: Sequence[TrajectoryPoint],
    validation_start_frame: int,
    panel_width: int,
) -> np.ndarray:
    """Render one side-by-side source and XZ-trajectory frame."""

    if source_bgr.ndim != 3 or source_bgr.shape[2] != 3 or source_bgr.dtype != np.uint8:
        raise ValueError("source frame must be uint8 BGR")
    source = source_bgr.copy()
    for segment in contiguous_segments(visible_points):
        pixels = np.asarray(
            [(round(point.u_px), round(point.v_px)) for point in segment],
            dtype=np.int32,
        )
        if len(pixels) >= 2:
            cv2.polylines(source, [pixels], False, (255, 180, 40), 5, cv2.LINE_AA)
            cv2.polylines(source, [pixels], False, (20, 50, 70), 1, cv2.LINE_AA)

    if current_observation is not None:
        pixels = current_observation.observation.landmark_pixels
        for start, end in pairwise(pixels):
            cv2.line(source, start, end, (70, 255, 180), 5, cv2.LINE_AA)
        for pixel in pixels[:-1]:
            cv2.circle(source, pixel, 8, (70, 255, 180), -1, cv2.LINE_AA)
        tip = pixels[-1]
        cv2.circle(source, tip, 14, (30, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(source, tip, 17, (255, 255, 255), 3, cv2.LINE_AA)
        _put_text_with_shadow(
            source,
            f"Predicted Z: {current_observation.trajectory_point.z_m * 100.0:.1f} cm",
            (36, 70),
            scale=1.0,
            color=(30, 220, 255),
            thickness=2,
        )
    else:
        _put_text_with_shadow(
            source,
            "No stored hand landmark for this frame",
            (36, 70),
            scale=0.8,
            color=(220, 220, 220),
            thickness=2,
        )
    _put_text_with_shadow(
        source,
        f"frame {frame_index}",
        (36, 112),
        scale=0.7,
        color=(240, 240, 240),
        thickness=2,
    )
    panel = _draw_xz_panel(
        height=source.shape[0],
        width=panel_width,
        visible_points=visible_points,
        all_points=all_points,
        frame_index=frame_index,
        validation_start_frame=validation_start_frame,
    )
    return np.concatenate((source, panel), axis=1)


# フレームごとの予測と注釈を動画ファイルに書き出します。
def render_demo_video(
    path: Path,
    *,
    inputs: DemoInputs,
    predictions: Sequence[PredictedObservation],
    panel_width: int,
    overwrite: bool,
    progress: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """Write the lossless source frames plus a synchronized trajectory plot."""

    if panel_width < 420:
        raise ValueError("panel_width must be at least 420 pixels")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing demo video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if temporary.exists():
        temporary.unlink()
    source_width = inputs.frames[0].width
    source_height = inputs.frames[0].height
    output_size = (source_width + panel_width, source_height)
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        inputs.fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {temporary}")

    predictions_by_frame = {
        prediction.observation.sample.frame_index: prediction for prediction in predictions
    }
    all_points = tuple(prediction.trajectory_point for prediction in predictions)
    visible_points: list[TrajectoryPoint] = []
    try:
        for frame in inputs.frames:
            source_bgr = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
            if source_bgr is None or source_bgr.shape != (frame.height, frame.width, 3):
                raise ValueError(
                    f"cached source frame is unreadable or has wrong shape: {frame.path}"
                )
            current = predictions_by_frame.get(frame.frame_index)
            if current is not None:
                visible_points.append(current.trajectory_point)
            rendered = render_demo_frame(
                source_bgr,
                frame_index=frame.frame_index,
                current_observation=current,
                visible_points=visible_points,
                all_points=all_points,
                validation_start_frame=inputs.validation_start_frame,
                panel_width=panel_width,
            )
            writer.write(rendered)
            if progress is not None and (
                (frame.frame_index + 1) % 100 == 0 or frame.frame_index + 1 == len(inputs.frames)
            ):
                progress(f"rendered {frame.frame_index + 1}/{len(inputs.frames)} frames")
    except BaseException:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("video writer produced no output")
    os.replace(temporary, path)
    return output_size


# 入力動画と学習済みモデルから、指先軌跡デモ一式を生成します。
def create_student_trajectory_demo(
    *,
    run_manifest_path: Path,
    output_dir: Path,
    device_name: str = "auto",
    batch_size: int = 32,
    panel_width: int = 720,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Create the complete minimal demo and return its output manifest."""

    output_dir = output_dir.resolve()
    video_path = output_dir / "demo.mp4"
    trajectory_path = output_dir / "trajectory.csv"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (video_path, trajectory_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing demo artifacts: {existing}")

    if progress is not None:
        progress("validating demo inputs")
    inputs = load_demo_inputs(run_manifest_path)
    if progress is not None:
        progress("loading Phase 8 student checkpoint")
    model, training_config, device, checkpoint = load_student_model(
        inputs,
        device_name=device_name,
    )
    if progress is not None:
        progress(f"running student inference on {len(inputs.observations)} identity frames")
    predictions = predict_trajectory(
        inputs,
        model=model,
        training_config=training_config,
        device=device,
        batch_size=batch_size,
        progress=progress,
    )
    points = tuple(prediction.trajectory_point for prediction in predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_trajectory_csv(trajectory_path, points)
    if progress is not None:
        progress(f"rendering {len(inputs.frames)} source frames")
    output_width, output_height = render_demo_video(
        video_path,
        inputs=inputs,
        predictions=predictions,
        panel_width=panel_width,
        overwrite=overwrite,
        progress=progress,
    )

    train_count = sum(prediction.observation.sample.split == "train" for prediction in predictions)
    validation_count = len(predictions) - train_count
    model_config = _require_mapping(checkpoint.get("model_config"), field="model_config")
    manifest: dict[str, Any] = {
        "format": DEMO_FORMAT,
        "format_version": DEMO_FORMAT_VERSION,
        "sequence_id": DEMO_SEQUENCE_ID,
        "purpose": "offline visualization demo",
        "ground_truth": False,
        "inputs": {
            "training_run_manifest": {
                "path": str(inputs.run_manifest_path),
                "sha256": inputs.run_manifest_sha256,
            },
            "student_checkpoint": {
                "path": str(inputs.checkpoint_path),
                "sha256": inputs.checkpoint_sha256,
                "epoch": int(checkpoint.get("epoch", -1)),
            },
            "student_dataset_manifest": {
                "path": str(inputs.dataset_manifest_path),
                "sha256": inputs.dataset_manifest_sha256,
            },
            "depth_pro_source_dataset_manifest": {
                "path": str(inputs.source_dataset_manifest_path),
                "sha256": inputs.source_dataset_manifest_sha256,
            },
            "source_video": {
                "path": str(inputs.source_video_path),
                "sha256": inputs.source_video_sha256,
            },
            "frame_cache_manifest": {
                "path": str(inputs.frame_cache_manifest_path),
                "sha256": inputs.frame_cache_manifest_sha256,
            },
            "landmarks": {
                "indices": [5, 6, 7, 8],
                "features": ["x_normalized", "y_normalized"],
                "z_mediapipe_relative_used": False,
            },
            "teacher_depth_used_for_inference": False,
        },
        "model": {
            "image_encoder_name": model_config.get("image_encoder_name"),
            "image_size": training_config.image_size,
            "resize_policy": "direct bicubic resize; no crop or letterbox",
            "precision": (
                "bfloat16"
                if training_config.precision == "bfloat16" and device.type == "cuda"
                else "float32"
            ),
            "device": str(device),
            "output": "single-frame optical-axis fingertip depth in metres",
        },
        "camera": {
            "intrinsics": inputs.intrinsics.as_dict(),
            "calibrated": False,
            "coordinate_convention": CAMERA_COORDINATE_CONVENTION,
        },
        "split_display": {
            "validation_start_frame_inclusive": inputs.validation_start_frame,
            "validation_label": "VALID tail (same video)",
        },
        "render": {
            "fps": inputs.fps,
            "codec": "mp4v",
            "source_size_px": {
                "width": inputs.frames[0].width,
                "height": inputs.frames[0].height,
            },
            "output_size_px": {"width": output_width, "height": output_height},
            "panel": "synchronized X-Z trajectory",
            "temporal_smoothing": False,
            "interpolation": False,
        },
        "counts": {
            "source_frames": len(inputs.frames),
            "predicted_identity_frames": len(predictions),
            "frames_without_selected_observation": len(inputs.frames) - len(predictions),
            "train_predictions": train_count,
            "validation_predictions": validation_count,
        },
        "prediction_depth_m": {
            "min": min(point.z_m for point in points),
            "max": max(point.z_m for point in points),
            "mean": float(np.mean([point.z_m for point in points], dtype=np.float64)),
        },
        "artifacts": {
            "demo_video": {
                "relative_path": video_path.name,
                "sha256": sha256_file(video_path),
                "size_bytes": video_path.stat().st_size,
            },
            "trajectory_csv": {
                "relative_path": trajectory_path.name,
                "sha256": sha256_file(trajectory_path),
            },
        },
        "notes": [
            "This is a visualization demo, not a ground-truth evaluation.",
            "Training-region predictions are in-sample; the final 7% is labelled as same-video validation.",
            "X/Y coordinates use an approximate centered-pinhole camera rather than calibrated intrinsics.",
        ],
    }
    write_json(manifest_path, manifest)
    if progress is not None:
        progress("demo artifacts complete")
    return manifest


__all__ = [
    "DEMO_FORMAT",
    "DEMO_FORMAT_VERSION",
    "DEMO_SEQUENCE_ID",
    "CachedFrame",
    "DemoInputs",
    "DemoObservation",
    "PredictedObservation",
    "create_student_trajectory_demo",
    "load_demo_inputs",
    "load_student_model",
    "predict_trajectory",
    "render_demo_frame",
    "render_demo_video",
]
