'TOMLなどの設定ファイルを読み込み、入力動画、保存先、前処理、教師、データセット、学習条件を型付き設定として検証します。'

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from fingertip_depth.student_model import StudentModelConfig
    from fingertip_depth.student_training import (
        StudentTrainingConfig,
        TeacherSpikeFilterConfig,
    )

SUPPORTED_VIDEO_EXTENSIONS = (".m4v", ".mov", ".mp4")
DEFAULT_LANDMARK_INDICES = (5, 6, 7, 8)
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)

TransferMode = Literal["copy", "hardlink"]


# 設定で明示されない場合に使うプロジェクト基準ディレクトリを決めます。
def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# 値が真偽値ではない数値型かどうかを判定します。
def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# 値を有限の浮動小数点数として検証・変換します。
def _finite_float(value: object, *, field: str) -> float:
    if not _is_number(value):
        raise TypeError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


# 値を正の有限浮動小数点数として検証・変換します。
def _positive_float(value: object, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


# 値が整数として有効か検証し、整数値に変換します。
def _integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


# 値を正の整数として検証・変換します。
def _positive_int(value: object, *, field: str) -> int:
    parsed = _integer(value, field=field)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


# 未指定を許容し、指定された場合は正の整数として検証します。
def _optional_positive_int(table: Mapping[str, Any], key: str, *, section: str) -> int | None:
    if key not in table:
        return None
    return _positive_int(table[key], field=f"{section}.{key}")


# 真偽値の設定を検証し、Pythonのboolとして返します。
def _boolean(table: Mapping[str, Any], key: str, default: bool, *, section: str) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{section}.{key} must be a boolean")
    return value


# 文字列の設定値を検証して返します。
def _string(table: Mapping[str, Any], key: str, default: str, *, section: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{section}.{key} must be a non-empty string")
    return value


# 設定ファイルの項目がテーブル形式か検証します。
def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a TOML table")
    return value


# 設定に許可されていないキーが含まれていないか検証します。
def _reject_unknown(table: Mapping[str, Any], allowed: set[str], *, section: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown {section} setting(s): {joined}")


# 相対パスをプロジェクト基準で解決し、絶対パスを返します。
def _resolve_path(value: object, *, project_root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


# プロジェクト基準の相対パスであることを検証・整形します。
def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


# フレーム転送方式が対応する値の一つか検証します。
def _transfer_mode(
    table: Mapping[str, Any],
    key: str,
    default: TransferMode,
    *,
    section: str,
) -> TransferMode:
    value = _string(table, key, default, section=section)
    if value not in {"copy", "hardlink"}:
        raise ValueError(f"{section}.{key} must be 'copy' or 'hardlink'")
    return value  # type: ignore[return-value]


# 整数の並びを検証してタプルへ変換します。
def _int_tuple(
    table: Mapping[str, Any],
    key: str,
    default: tuple[int, ...],
    *,
    section: str,
) -> tuple[int, ...]:
    value = table.get(key, list(default))
    if not isinstance(value, list) or not value:
        raise TypeError(f"{section}.{key} must be a non-empty integer array")
    result = tuple(_integer(item, field=f"{section}.{key}") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{section}.{key} must contain unique values")
    return result


# 3要素の有限浮動小数点数として検証・変換します。
def _float_triplet(
    table: Mapping[str, Any],
    key: str,
    default: tuple[float, float, float],
    *,
    section: str,
) -> tuple[float, float, float]:
    value = table.get(key, list(default))
    if not isinstance(value, list) or len(value) != 3:
        raise TypeError(f"{section}.{key} must be a three-number array")
    parsed = tuple(_finite_float(item, field=f"{section}.{key}") for item in value)
    return parsed  # type: ignore[return-value]


# 一動画に限ってパイプライン共通設定を上書きする値を保持します。
@dataclass(frozen=True, slots=True)
class VideoOverrideConfig:
    """Per-video values that differ from the pipeline defaults."""

    focal_35mm_mm: float

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        _positive_float(self.focal_35mm_mm, field="video override focal_35mm_mm")


# 動画などパイプラインの入力場所を保持します。
@dataclass(frozen=True, slots=True)
class InputConfig:
    train_dir: Path
    validation_dir: Path
    extensions: tuple[str, ...]
    default_focal_35mm_mm: float
    video_overrides: Mapping[Path, VideoOverrideConfig]

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if not self.extensions:
            raise ValueError("input.extensions must not be empty")
        if any(extension not in SUPPORTED_VIDEO_EXTENSIONS for extension in self.extensions):
            raise ValueError("input.extensions contains an unsupported video extension")
        _positive_float(self.default_focal_35mm_mm, field="input.default_focal_35mm_mm")


# 前処理、データセット、学習runの保存先を保持します。
@dataclass(frozen=True, slots=True)
class OutputConfig:
    processed_dir: Path
    dataset_dir: Path
    runs_dir: Path


# フレーム抽出と手ランドマーク検出の条件を保持します。
@dataclass(frozen=True, slots=True)
class PrepareConfig:
    hand_model_path: Path
    landmark_indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES
    frame_transfer_mode: TransferMode = "hardlink"
    max_frames: int | None = None

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if not self.landmark_indices or len(set(self.landmark_indices)) != len(
            self.landmark_indices
        ):
            raise ValueError("prepare.landmark_indices must be non-empty and unique")
        if any(index < 0 or index > 20 for index in self.landmark_indices):
            raise ValueError("prepare.landmark_indices values must be in [0, 20]")
        if self.frame_transfer_mode not in {"copy", "hardlink"}:
            raise ValueError("prepare.frame_transfer_mode must be copy or hardlink")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("prepare.max_frames must be positive")


# 教師モデル、実行環境、疑似ラベル対象の選択条件を保持します。
@dataclass(frozen=True, slots=True)
class TeacherConfig:
    depth_pro_project: Path
    device: str = "cuda:0"
    frame_transfer_mode: TransferMode = "hardlink"
    max_frames: int | None = None
    checkpoint_interval_frames: int = 100
    teacher_selection_report: Path | None = None

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("teacher.device must not be empty")
        if self.frame_transfer_mode not in {"copy", "hardlink"}:
            raise ValueError("teacher.frame_transfer_mode must be copy or hardlink")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("teacher.max_frames must be positive")
        if isinstance(self.checkpoint_interval_frames, bool) or not isinstance(
            self.checkpoint_interval_frames, int
        ):
            raise TypeError("teacher.checkpoint_interval_frames must be an integer")
        if self.checkpoint_interval_frames <= 0:
            raise ValueError("teacher.checkpoint_interval_frames must be positive")


# データセットの統合、分割、左右反転に関する条件を保持します。
@dataclass(frozen=True, slots=True)
class DatasetConfig:
    frame_transfer_mode: TransferMode = "hardlink"
    validation_tail_fraction: float | None = None

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if self.frame_transfer_mode not in {"copy", "hardlink"}:
            raise ValueError("dataset.frame_transfer_mode must be copy or hardlink")
        if self.validation_tail_fraction is not None and not (
            0.0 < self.validation_tail_fraction < 1.0
        ):
            raise ValueError("dataset.validation_tail_fraction must be in (0, 1)")


# 生徒モデルの構造と入力形式の設定です。
@dataclass(frozen=True, slots=True)
class StudentModelSettings:
    image_encoder_name: str = "vit_small_patch16_224.dino"
    pretrained_image_encoder: bool = True
    landmark_indices: tuple[int, ...] = DEFAULT_LANDMARK_INDICES
    fusion_layers: int = 2
    fusion_heads: int = 6
    fusion_mlp_ratio: float = 4.0
    dropout: float = 0.1

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if not self.image_encoder_name:
            raise ValueError("training.model.image_encoder_name must not be empty")
        if not self.landmark_indices or len(set(self.landmark_indices)) != len(
            self.landmark_indices
        ):
            raise ValueError("training.model.landmark_indices must be non-empty and unique")
        if any(index < 0 or index > 20 for index in self.landmark_indices):
            raise ValueError("training.model.landmark_indices values must be in [0, 20]")
        if self.fusion_layers <= 0 or self.fusion_heads <= 0:
            raise ValueError("training.model fusion layers and heads must be positive")
        if self.fusion_mlp_ratio <= 0.0:
            raise ValueError("training.model.fusion_mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("training.model.dropout must be in [0, 1)")


# 最適化器、学習率、重み減衰などの設定です。
@dataclass(frozen=True, slots=True)
class StudentOptimizerSettings:
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
            raise ValueError("training optimizer epochs and batch_size must be positive")
        if self.encoder_learning_rate <= 0.0 or self.head_learning_rate <= 0.0:
            raise ValueError("training optimizer learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("training.optimizer.weight_decay must be non-negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("training.optimizer.warmup_fraction must be in [0, 1)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("training.optimizer.gradient_clip_norm must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("training.optimizer.early_stopping_patience must be non-negative")
        if self.image_size <= 0 or self.num_workers < 0:
            raise ValueError("training image_size must be positive and num_workers non-negative")
        if any(value <= 0.0 for value in self.image_std):
            raise ValueError("training.optimizer.image_std values must be positive")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("training.optimizer.precision must be float32 or bfloat16")


# 教師深度のスパイクと上限値による除外設定です。
@dataclass(frozen=True, slots=True)
class SpikeFilterSettings:
    enabled: bool = True
    frame_radius: int = 3
    max_frame_gap: int = 1
    min_neighbors: int = 3
    absolute_floor_m: float = 0.15
    relative_floor_fraction: float = 0.50
    mad_multiplier: float = 6.0
    mad_scale: float = 1.4826
    max_depth_m: float | None = None

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if self.frame_radius <= 0 or self.max_frame_gap <= 0:
            raise ValueError("training spike-filter radii must be positive")
        if self.min_neighbors <= 0 or self.min_neighbors > 2 * self.frame_radius:
            raise ValueError("training spike-filter min_neighbors is outside the valid range")
        thresholds = (
            self.absolute_floor_m,
            self.relative_floor_fraction,
            self.mad_multiplier,
            self.mad_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
            raise ValueError("training spike-filter thresholds must be finite and positive")
        if self.max_depth_m is not None and (
            not math.isfinite(self.max_depth_m) or self.max_depth_m <= 0.0
        ):
            raise ValueError("training spike-filter max_depth_m must be finite and positive")


# モデル、最適化、学習回数、評価に関する設定をまとめます。
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    device: str
    model: StudentModelSettings
    optimizer: StudentOptimizerSettings
    spike_filter: SpikeFilterSettings

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("training.device must not be empty")

    # 設定値からStudentModelConfigを作成して返します。
    def model_config(self) -> StudentModelConfig:
        """Convert the loaded values to the existing model configuration."""

        from fingertip_depth.student_model import StudentModelConfig

        return StudentModelConfig(**asdict(self.model))

    # 設定値からStudentTrainingConfigを作成して返します。
    def training_config(self) -> StudentTrainingConfig:
        """Convert the loaded values to the existing optimizer configuration."""

        from fingertip_depth.student_training import StudentTrainingConfig

        return StudentTrainingConfig(**asdict(self.optimizer))

    # 設定値からTeacherSpikeFilterConfigを作成して返します。
    def spike_filter_config(self) -> TeacherSpikeFilterConfig:
        """Convert the loaded values to the existing teacher-spike configuration."""

        from fingertip_depth.student_training import TeacherSpikeFilterConfig

        return TeacherSpikeFilterConfig(**asdict(self.spike_filter))


# 動画探索から学習までの設定とプロジェクト基準パスをまとめます。
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    project_root: Path
    config_path: Path
    input: InputConfig
    output: OutputConfig
    prepare: PrepareConfig
    teacher: TeacherConfig
    dataset: DatasetConfig
    training: TrainingConfig

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if self.input.train_dir == self.input.validation_dir:
            raise ValueError("input train and validation directories must be different")
        output_dirs = (
            self.output.processed_dir,
            self.output.dataset_dir,
            self.output.runs_dir,
        )
        if len(set(output_dirs)) != len(output_dirs):
            raise ValueError("output processed, dataset and runs directories must be different")
        for output_dir in output_dirs:
            for input_dir in (self.input.train_dir, self.input.validation_dir):
                if output_dir.is_relative_to(input_dir):
                    raise ValueError(
                        f"output directory must not be inside an input directory: {output_dir}"
                    )
        missing_landmarks = sorted(
            set(self.training.model.landmark_indices) - set(self.prepare.landmark_indices)
        )
        if missing_landmarks:
            raise ValueError(
                "training.model.landmark_indices must be included in prepare.landmark_indices: "
                f"missing {missing_landmarks}"
            )

    # 主要なフィールドを、JSONへ保存できる辞書に変換します。
    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic, project-relative payload suitable for hashing."""

        root = self.project_root
        overrides = {
            _relative_path(path, root): {"focal_35mm_mm": override.focal_35mm_mm}
            for path, override in sorted(
                self.input.video_overrides.items(),
                key=lambda item: _relative_path(item[0], root),
            )
        }
        optimizer = asdict(self.training.optimizer)
        optimizer["image_mean"] = list(self.training.optimizer.image_mean)
        optimizer["image_std"] = list(self.training.optimizer.image_std)
        model = asdict(self.training.model)
        model["landmark_indices"] = list(self.training.model.landmark_indices)
        return {
            "input": {
                "train_dir": _relative_path(self.input.train_dir, root),
                "validation_dir": _relative_path(self.input.validation_dir, root),
                "extensions": list(self.input.extensions),
                "default_focal_35mm_mm": self.input.default_focal_35mm_mm,
            },
            "output": {
                "processed_dir": _relative_path(self.output.processed_dir, root),
                "dataset_dir": _relative_path(self.output.dataset_dir, root),
                "runs_dir": _relative_path(self.output.runs_dir, root),
            },
            "video_overrides": overrides,
            "prepare": {
                "hand_model_path": _relative_path(self.prepare.hand_model_path, root),
                "landmark_indices": list(self.prepare.landmark_indices),
                "frame_transfer_mode": self.prepare.frame_transfer_mode,
                "max_frames": self.prepare.max_frames,
            },
            "teacher": {
                "depth_pro_project": _relative_path(self.teacher.depth_pro_project, root),
                "device": self.teacher.device,
                "frame_transfer_mode": self.teacher.frame_transfer_mode,
                "max_frames": self.teacher.max_frames,
                "checkpoint_interval_frames": self.teacher.checkpoint_interval_frames,
                "teacher_selection_report": (
                    None
                    if self.teacher.teacher_selection_report is None
                    else _relative_path(self.teacher.teacher_selection_report, root)
                ),
            },
            "dataset": {
                "frame_transfer_mode": self.dataset.frame_transfer_mode,
                "validation_tail_fraction": self.dataset.validation_tail_fraction,
            },
            "training": {
                "device": self.training.device,
                "model": model,
                "optimizer": optimizer,
                "spike_filter": asdict(self.training.spike_filter),
            },
        }

    # キャッシュキーに使う設定項目だけを安定した辞書にまとめます。
    def fingerprint_dict(self) -> dict[str, Any]:
        """Alias with an explicit name for callers constructing a content fingerprint."""

        return self.as_dict()


# 入力テーブルを検証し、動画ディレクトリなどのInputConfigを作ります。
def _load_input(
    table: Mapping[str, Any],
    overrides_table: Mapping[str, Any],
    *,
    project_root: Path,
) -> InputConfig:
    _reject_unknown(
        table,
        {"train_dir", "validation_dir", "extensions", "default_focal_35mm_mm"},
        section="input",
    )
    train_dir = _resolve_path(
        table.get("train_dir", "data/training_videos/train"),
        project_root=project_root,
        field="input.train_dir",
    )
    validation_dir = _resolve_path(
        table.get("validation_dir", "data/training_videos/validation"),
        project_root=project_root,
        field="input.validation_dir",
    )
    raw_extensions = table.get("extensions", list(SUPPORTED_VIDEO_EXTENSIONS))
    if not isinstance(raw_extensions, list) or not raw_extensions:
        raise TypeError("input.extensions must be a non-empty string array")
    normalized_extensions: set[str] = set()
    for raw_extension in raw_extensions:
        if not isinstance(raw_extension, str) or not raw_extension.strip():
            raise TypeError("input.extensions must contain non-empty strings")
        extension = raw_extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported input video extension: {extension}")
        normalized_extensions.add(extension)
    default_focal = _positive_float(
        table.get("default_focal_35mm_mm", 36.0),
        field="input.default_focal_35mm_mm",
    )

    overrides: dict[Path, VideoOverrideConfig] = {}
    for raw_path, raw_override in overrides_table.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise TypeError("video_overrides keys must be non-empty paths")
        if Path(raw_path).is_absolute():
            raise ValueError("video_overrides keys must be project-root-relative paths")
        if not isinstance(raw_override, dict):
            raise TypeError(f"video_overrides.{raw_path} must be a TOML table")
        _reject_unknown(raw_override, {"focal_35mm_mm"}, section=f"video_overrides.{raw_path}")
        if "focal_35mm_mm" not in raw_override:
            raise ValueError(f"video_overrides.{raw_path}.focal_35mm_mm is required")
        override_path = _resolve_path(
            raw_path,
            project_root=project_root,
            field=f"video_overrides.{raw_path}",
        )
        try:
            override_path.relative_to(project_root)
        except ValueError as error:
            raise ValueError("video_overrides paths must stay inside the project root") from error
        if not (
            override_path.is_relative_to(train_dir) or override_path.is_relative_to(validation_dir)
        ):
            raise ValueError(
                f"video override is outside input train/validation directories: {raw_path}"
            )
        overrides[override_path] = VideoOverrideConfig(
            focal_35mm_mm=_positive_float(
                raw_override["focal_35mm_mm"],
                field=f"video_overrides.{raw_path}.focal_35mm_mm",
            )
        )
    return InputConfig(
        train_dir=train_dir,
        validation_dir=validation_dir,
        extensions=tuple(sorted(normalized_extensions)),
        default_focal_35mm_mm=default_focal,
        video_overrides=MappingProxyType(overrides),
    )


# 出力テーブルを検証し、各成果物の保存先を解決したOutputConfigを作ります。
def _load_output(table: Mapping[str, Any], *, project_root: Path) -> OutputConfig:
    _reject_unknown(
        table,
        {"processed_dir", "dataset_dir", "runs_dir"},
        section="output",
    )
    return OutputConfig(
        processed_dir=_resolve_path(
            table.get("processed_dir", "outputs/training_pipeline/processed"),
            project_root=project_root,
            field="output.processed_dir",
        ),
        dataset_dir=_resolve_path(
            table.get("dataset_dir", "outputs/training_pipeline/dataset"),
            project_root=project_root,
            field="output.dataset_dir",
        ),
        runs_dir=_resolve_path(
            table.get("runs_dir", "outputs/training_pipeline/runs"),
            project_root=project_root,
            field="output.runs_dir",
        ),
    )


# 前処理テーブルから手モデル、ランドマーク、フレーム上限などのPrepareConfigを作ります。
def _load_prepare(table: Mapping[str, Any], *, project_root: Path) -> PrepareConfig:
    _reject_unknown(
        table,
        {"hand_model_path", "landmark_indices", "frame_transfer_mode", "max_frames"},
        section="prepare",
    )
    return PrepareConfig(
        hand_model_path=_resolve_path(
            table.get("hand_model_path", "assets/hand_landmarker.task"),
            project_root=project_root,
            field="prepare.hand_model_path",
        ),
        landmark_indices=_int_tuple(
            table,
            "landmark_indices",
            DEFAULT_LANDMARK_INDICES,
            section="prepare",
        ),
        frame_transfer_mode=_transfer_mode(
            table,
            "frame_transfer_mode",
            "hardlink",
            section="prepare",
        ),
        max_frames=_optional_positive_int(table, "max_frames", section="prepare"),
    )


# 教師テーブルからDepth Pro環境、実行デバイス、選択レポートなどのTeacherConfigを作ります。
def _load_teacher(table: Mapping[str, Any], *, project_root: Path) -> TeacherConfig:
    _reject_unknown(
        table,
        {
            "depth_pro_project",
            "device",
            "frame_transfer_mode",
            "max_frames",
            "checkpoint_interval_frames",
            "teacher_selection_report",
        },
        section="teacher",
    )
    raw_report = table.get("teacher_selection_report")
    report = (
        None
        if raw_report is None
        else _resolve_path(
            raw_report,
            project_root=project_root,
            field="teacher.teacher_selection_report",
        )
    )
    return TeacherConfig(
        depth_pro_project=_resolve_path(
            table.get("depth_pro_project", "environments/depth_pro"),
            project_root=project_root,
            field="teacher.depth_pro_project",
        ),
        device=_string(table, "device", "cuda:0", section="teacher"),
        frame_transfer_mode=_transfer_mode(
            table,
            "frame_transfer_mode",
            "hardlink",
            section="teacher",
        ),
        max_frames=_optional_positive_int(table, "max_frames", section="teacher"),
        checkpoint_interval_frames=_positive_int(
            table.get("checkpoint_interval_frames", 100),
            field="teacher.checkpoint_interval_frames",
        ),
        teacher_selection_report=report,
    )


# データセットテーブルから統合・分割・反転条件のDatasetConfigを作ります。
def _load_dataset(table: Mapping[str, Any]) -> DatasetConfig:
    _reject_unknown(
        table,
        {"frame_transfer_mode", "validation_tail_fraction"},
        section="dataset",
    )
    raw_fraction = table.get("validation_tail_fraction")
    fraction = (
        None
        if raw_fraction is None
        else _finite_float(raw_fraction, field="dataset.validation_tail_fraction")
    )
    return DatasetConfig(
        frame_transfer_mode=_transfer_mode(
            table,
            "frame_transfer_mode",
            "hardlink",
            section="dataset",
        ),
        validation_tail_fraction=fraction,
    )


# モデルテーブルからViTとランドマークTransformerのStudentModelSettingsを作ります。
def _load_model(table: Mapping[str, Any]) -> StudentModelSettings:
    _reject_unknown(
        table,
        {
            "image_encoder_name",
            "pretrained_image_encoder",
            "landmark_indices",
            "fusion_layers",
            "fusion_heads",
            "fusion_mlp_ratio",
            "dropout",
        },
        section="training.model",
    )
    return StudentModelSettings(
        image_encoder_name=_string(
            table,
            "image_encoder_name",
            "vit_small_patch16_224.dino",
            section="training.model",
        ),
        pretrained_image_encoder=_boolean(
            table,
            "pretrained_image_encoder",
            True,
            section="training.model",
        ),
        landmark_indices=_int_tuple(
            table,
            "landmark_indices",
            DEFAULT_LANDMARK_INDICES,
            section="training.model",
        ),
        fusion_layers=_positive_int(
            table.get("fusion_layers", 2), field="training.model.fusion_layers"
        ),
        fusion_heads=_positive_int(
            table.get("fusion_heads", 6), field="training.model.fusion_heads"
        ),
        fusion_mlp_ratio=_positive_float(
            table.get("fusion_mlp_ratio", 4.0),
            field="training.model.fusion_mlp_ratio",
        ),
        dropout=_finite_float(table.get("dropout", 0.1), field="training.model.dropout"),
    )


# 最適化テーブルから学習率、バッチサイズ、エポックなどの設定を作ります。
def _load_optimizer(table: Mapping[str, Any]) -> StudentOptimizerSettings:
    allowed = {
        "epochs",
        "batch_size",
        "encoder_learning_rate",
        "head_learning_rate",
        "weight_decay",
        "warmup_fraction",
        "gradient_clip_norm",
        "early_stopping_patience",
        "image_size",
        "image_mean",
        "image_std",
        "num_workers",
        "seed",
        "precision",
        "freeze_image_encoder",
        "preload_images",
        "verify_image_png_sha256",
    }
    _reject_unknown(table, allowed, section="training.optimizer")
    precision = _string(table, "precision", "bfloat16", section="training.optimizer")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("training.optimizer.precision must be float32 or bfloat16")
    return StudentOptimizerSettings(
        epochs=_positive_int(table.get("epochs", 20), field="training.optimizer.epochs"),
        batch_size=_positive_int(
            table.get("batch_size", 32), field="training.optimizer.batch_size"
        ),
        encoder_learning_rate=_positive_float(
            table.get("encoder_learning_rate", 1e-5),
            field="training.optimizer.encoder_learning_rate",
        ),
        head_learning_rate=_positive_float(
            table.get("head_learning_rate", 1e-4),
            field="training.optimizer.head_learning_rate",
        ),
        weight_decay=_finite_float(
            table.get("weight_decay", 0.05), field="training.optimizer.weight_decay"
        ),
        warmup_fraction=_finite_float(
            table.get("warmup_fraction", 0.05),
            field="training.optimizer.warmup_fraction",
        ),
        gradient_clip_norm=_positive_float(
            table.get("gradient_clip_norm", 1.0),
            field="training.optimizer.gradient_clip_norm",
        ),
        early_stopping_patience=_integer(
            table.get("early_stopping_patience", 6),
            field="training.optimizer.early_stopping_patience",
        ),
        image_size=_positive_int(
            table.get("image_size", 224), field="training.optimizer.image_size"
        ),
        image_mean=_float_triplet(
            table,
            "image_mean",
            DEFAULT_IMAGE_MEAN,
            section="training.optimizer",
        ),
        image_std=_float_triplet(
            table,
            "image_std",
            DEFAULT_IMAGE_STD,
            section="training.optimizer",
        ),
        num_workers=_integer(table.get("num_workers", 4), field="training.optimizer.num_workers"),
        seed=_integer(table.get("seed", 20260925), field="training.optimizer.seed"),
        precision=precision,  # type: ignore[arg-type]
        freeze_image_encoder=_boolean(
            table,
            "freeze_image_encoder",
            False,
            section="training.optimizer",
        ),
        preload_images=_boolean(
            table,
            "preload_images",
            True,
            section="training.optimizer",
        ),
        verify_image_png_sha256=_boolean(
            table,
            "verify_image_png_sha256",
            True,
            section="training.optimizer",
        ),
    )


# スパイク除外テーブルを検証し、教師深度の外れ値除外設定を作ります。
def _load_spike_filter(table: Mapping[str, Any]) -> SpikeFilterSettings:
    allowed = {
        "enabled",
        "frame_radius",
        "max_frame_gap",
        "min_neighbors",
        "absolute_floor_m",
        "relative_floor_fraction",
        "mad_multiplier",
        "mad_scale",
        "max_depth_m",
    }
    _reject_unknown(table, allowed, section="training.spike_filter")
    return SpikeFilterSettings(
        enabled=_boolean(table, "enabled", True, section="training.spike_filter"),
        frame_radius=_positive_int(
            table.get("frame_radius", 3), field="training.spike_filter.frame_radius"
        ),
        max_frame_gap=_positive_int(
            table.get("max_frame_gap", 1), field="training.spike_filter.max_frame_gap"
        ),
        min_neighbors=_positive_int(
            table.get("min_neighbors", 3), field="training.spike_filter.min_neighbors"
        ),
        absolute_floor_m=_positive_float(
            table.get("absolute_floor_m", 0.15),
            field="training.spike_filter.absolute_floor_m",
        ),
        relative_floor_fraction=_positive_float(
            table.get("relative_floor_fraction", 0.50),
            field="training.spike_filter.relative_floor_fraction",
        ),
        mad_multiplier=_positive_float(
            table.get("mad_multiplier", 6.0),
            field="training.spike_filter.mad_multiplier",
        ),
        mad_scale=_positive_float(
            table.get("mad_scale", 1.4826), field="training.spike_filter.mad_scale"
        ),
        max_depth_m=(
            _positive_float(table["max_depth_m"], field="training.spike_filter.max_depth_m")
            if "max_depth_m" in table
            else None
        ),
    )


# 学習テーブルを検証し、モデル・最適化・評価条件をまとめたTrainingConfigを作ります。
def _load_training(table: Mapping[str, Any]) -> TrainingConfig:
    _reject_unknown(table, {"device", "model", "optimizer", "spike_filter"}, section="training")
    return TrainingConfig(
        device=_string(table, "device", "cuda:0", section="training"),
        model=_load_model(_table(table, "model")),
        optimizer=_load_optimizer(_table(table, "optimizer")),
        spike_filter=_load_spike_filter(_table(table, "spike_filter")),
    )


# TOML設定ファイルを読み、各工程の型付き設定を構築します。
def load_pipeline_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> PipelineConfig:
    """Load TOML, resolving every relative path against the project root.

    Relative paths deliberately do not use the TOML file's parent directory. This keeps the
    configuration stable when it is moved within the repository.
    """

    root = (
        _default_project_root()
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    with config_path.open("rb") as file:
        data = tomllib.load(file)
    if not isinstance(data, dict):
        raise TypeError("pipeline TOML root must be a table")
    _reject_unknown(
        data,
        {"input", "output", "video_overrides", "prepare", "teacher", "dataset", "training"},
        section="top-level",
    )
    return PipelineConfig(
        project_root=root,
        config_path=config_path,
        input=_load_input(
            _table(data, "input"),
            _table(data, "video_overrides"),
            project_root=root,
        ),
        output=_load_output(_table(data, "output"), project_root=root),
        prepare=_load_prepare(_table(data, "prepare"), project_root=root),
        teacher=_load_teacher(_table(data, "teacher"), project_root=root),
        dataset=_load_dataset(_table(data, "dataset")),
        training=_load_training(_table(data, "training")),
    )


__all__ = [
    "SUPPORTED_VIDEO_EXTENSIONS",
    "DatasetConfig",
    "InputConfig",
    "OutputConfig",
    "PipelineConfig",
    "PrepareConfig",
    "SpikeFilterSettings",
    "StudentModelSettings",
    "StudentOptimizerSettings",
    "TeacherConfig",
    "TrainingConfig",
    "VideoOverrideConfig",
    "load_pipeline_config",
]
