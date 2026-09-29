'学習済み手指深度モデルをCore MLへ変換し、変換対象・計算グラフ・出力の整合性を記録して監査可能にします。'

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .student_model import FingertipDepthStudent, StudentModelConfig
from .student_training import (
    STUDENT_CHECKPOINT_FORMAT_VERSION,
    StudentTrainingConfig,
)

COREML_EXPORT_FORMAT = "fingertip-depth-student-coreml-export"
COREML_EXPORT_FORMAT_VERSION = 2
CHECKPOINT_FORMAT = "fingertip-depth-student-checkpoint"
TRAINING_RUN_FORMAT = "fingertip-depth-student-training"
TRAINING_RUN_FORMAT_VERSION = 2
EXPECTED_LANDMARK_INDICES = (5, 6, 7, 8)
EXPECTED_IMAGE_SIZE = 224
IMAGE_INPUT_NAME = "image"
LANDMARK_INPUT_NAME = "landmarks_xy"
OUTPUT_NAME = "depth_m"
TORCH_EXPORT_BACKEND = "torch.export"
TORCHSCRIPT_BACKEND = "torchscript"
FORBIDDEN_TORCH_EXPORT_OPERATOR_PREFIXES = (
    "aten._transformer_encoder_layer_fwd",
    "aten.Int",
)
TARGET_DEVICE = {
    "marketing_name": "iPhone 15",
    "hardware_identifier": "iPhone15,4",
    "operating_system": "iOS 26.6.1",
}


# Core MLへ変換する学習済みモデルと、その出所を検証する情報を保持します。
@dataclass(frozen=True, slots=True)
class StudentExportSource:
    """Verified checkpoint and reconstructed inference model."""

    run_manifest_path: Path
    run_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    checkpoint: Mapping[str, Any]
    model: FingertipDepthStudent
    model_config: StudentModelConfig
    training_config: StudentTrainingConfig


# 学習済み生徒モデルをCore ML変換で扱える入出力形式に包みます。
class StudentCoreMLWrapper(nn.Module):
    """Expose deployment-friendly raw RGB and normalized landmark inputs.

    Core ML's image input scales uint8 RGB pixels by ``1 / 255``. Therefore
    ``image_rgb_0_1`` is already in [0, 1] when this module is executed. The
    wrapper owns the remaining ImageNet normalization and the landmark
    centering so the Swift application cannot accidentally omit or duplicate
    either operation.
    """

    # 必要な引数を検証し、インスタンスの状態を初期化します。
    def __init__(
        self,
        model: FingertipDepthStudent,
        *,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ) -> None:
        super().__init__()
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("image mean and std must each contain three values")
        if any(float(value) <= 0.0 for value in image_std):
            raise ValueError("image std values must be positive")
        self.model = model
        self.register_buffer(
            "image_mean",
            torch.tensor(tuple(image_mean), dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(tuple(image_std), dtype=torch.float32).view(1, 3, 1, 1),
        )

    # 入力をニューラルネットワークに通し、予測値を返します。
    def forward(
        self,
        image_rgb_0_1: torch.Tensor,
        landmarks_xy_0_1: torch.Tensor,
    ) -> torch.Tensor:
        normalized_image = (image_rgb_0_1 - self.image_mean) / self.image_std
        centered_landmarks = landmarks_xy_0_1 * 2.0 - 1.0
        # [batch, 1]の出力にすると、Core MLのMultiArray出力が一つに固定されます。
        return self.model(normalized_image, centered_landmarks).unsqueeze(1)


# 固定形状の入力を使ってCore ML変換用のグラフをトレースするラッパーです。
class CoreMLTraceWrapper(StudentCoreMLWrapper):
    """Fixed-batch equivalent that avoids unsupported fused PyTorch operators.

    PyTorch's evaluation fast path emits ``_transformer_encoder_layer_fwd``
    and dynamic ``aten::Int`` nodes that coremltools 9 cannot convert. This
    wrapper evaluates the same trained weights with explicit matrix multiply,
    softmax, residual, MLP, and normalization operations. It is intentionally
    fixed to the Core ML deployment shape (batch 1, 224x224, four landmarks).
    """

    IMAGE_TOKEN_COUNT = 197
    FUSION_TOKEN_COUNT = 202

    # 必要な引数を検証し、インスタンスの状態を初期化します。
    def __init__(
        self,
        model: FingertipDepthStudent,
        *,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ) -> None:
        super().__init__(model, image_mean=image_mean, image_std=image_std)
        self.embedding_dim = int(model.embedding_dim)
        self.head_count = int(model.config.fusion_heads)
        self.head_dim = self.embedding_dim // self.head_count
        if self.embedding_dim != 384 or self.head_count != 6 or self.head_dim != 64:
            raise ValueError("Core ML MVP requires the trained 384-dim, six-head student")

    # 画像・固定・バッチを指定形式に符号化します。
    def _encode_image_fixed_batch(self, image: torch.Tensor) -> torch.Tensor:
        encoder = self.model.image_encoder
        tokens = encoder.patch_embed(image)
        # cls_tokenにはすでにバッチ次元があります。expand()を避けることで、トレース時の
        # Tensorから整数への動的変換を計算グラフに含めずに済みます。
        tokens = torch.cat((encoder.cls_token, tokens), dim=1)
        tokens = encoder.pos_drop(tokens + encoder.pos_embed)
        tokens = encoder.patch_drop(tokens)
        tokens = encoder.norm_pre(tokens)
        tokens = encoder.blocks(tokens)
        return encoder.norm(tokens)

    # Transformer層の自己注意計算を明示的な行列演算で実装し、変換可能なTensor出力を返します。
    def _self_attention(
        self,
        inputs: torch.Tensor,
        layer: nn.TransformerEncoderLayer,
    ) -> torch.Tensor:
        attention = layer.self_attn
        qkv = F.linear(inputs, attention.in_proj_weight, attention.in_proj_bias)
        query, key, value = torch.split(qkv, self.embedding_dim, dim=-1)
        query = query.reshape(1, self.FUSION_TOKEN_COUNT, self.head_count, self.head_dim).transpose(
            1, 2
        )
        key = key.reshape(1, self.FUSION_TOKEN_COUNT, self.head_count, self.head_dim).transpose(
            1, 2
        )
        value = value.reshape(1, self.FUSION_TOKEN_COUNT, self.head_count, self.head_dim).transpose(
            1, 2
        )
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim**-0.5),
            dim=-1,
        )
        attended = (
            torch.matmul(weights, value)
            .transpose(1, 2)
            .reshape(1, self.FUSION_TOKEN_COUNT, self.embedding_dim)
        )
        return F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)

    # 入力をニューラルネットワークに通し、予測値を返します。
    def forward(
        self,
        image_rgb_0_1: torch.Tensor,
        landmarks_xy_0_1: torch.Tensor,
    ) -> torch.Tensor:
        image = (image_rgb_0_1 - self.image_mean) / self.image_std
        landmarks = landmarks_xy_0_1 * 2.0 - 1.0
        image_tokens = self._encode_image_fixed_batch(image)
        image_tokens = image_tokens + self.model.image_modality_token
        coordinate_tokens = self.model.landmark_coordinate_mlp(landmarks)
        type_tokens = self.model.landmark_type_tokens().unsqueeze(0)
        landmark_tokens = coordinate_tokens + type_tokens + self.model.landmark_modality_token
        depth_query = self.model.depth_query + self.model.query_modality_token
        encoded = torch.cat((depth_query, image_tokens, landmark_tokens), dim=1)
        for layer in self.model.fusion_transformer.layers:
            encoded = encoded + self._self_attention(layer.norm1(encoded), layer)
            normalized = layer.norm2(encoded)
            encoded = encoded + layer.linear2(F.gelu(layer.linear1(normalized)))
        encoded = self.model.fusion_transformer.norm(encoded)
        return self.model.depth_head(encoded[:, 0]).reshape(1, 1)


# 指定したファイルの内容からSHA-256を計算します。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


# 成果物の相対パスがマニフェストの基準ディレクトリ内にあり、実ファイルが存在することを確認します。
def _contained_artifact(root: Path, relative_path: object, *, field: str) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to its manifest")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} escapes its manifest directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


# チェックポイントのモデル設定からStudentModelConfigを復元し、学習済み重みのロード時に事前学習重みを取得しないようにします。
def _model_config(raw: object) -> StudentModelConfig:
    config = _require_mapping(raw, field="checkpoint.model_config")
    allowed = {field.name for field in fields(StudentModelConfig)}
    values = {name: config[name] for name in allowed if name in config}
    # 学習済み重みを復元するときに、ネットワークから重みを取得しないようにします。
    values["pretrained_image_encoder"] = False
    if "landmark_indices" in values:
        values["landmark_indices"] = tuple(int(value) for value in values["landmark_indices"])
    return StudentModelConfig(**values)


# チェックポイントの学習設定を検証し、タプル項目を復元してStudentTrainingConfigを作ります。
def _training_config(raw: object) -> StudentTrainingConfig:
    config = _require_mapping(raw, field="checkpoint.training_config")
    allowed = {field.name for field in fields(StudentTrainingConfig)}
    values = {name: config[name] for name in allowed if name in config}
    if "image_mean" in values:
        values["image_mean"] = tuple(float(value) for value in values["image_mean"])
    if "image_std" in values:
        values["image_std"] = tuple(float(value) for value in values["image_std"])
    return StudentTrainingConfig(**values)


# runマニフェストと最良チェックポイントのハッシュ・形式・データセット参照を検証し、学習済みモデルを復元します。
def load_student_export_source(run_manifest_path: Path) -> StudentExportSource:
    """Verify the selected training run and reconstruct its best checkpoint."""

    run_manifest_path = run_manifest_path.resolve()
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    run_manifest = _read_json(run_manifest_path)
    if (
        run_manifest.get("format") != TRAINING_RUN_FORMAT
        or run_manifest.get("format_version") != TRAINING_RUN_FORMAT_VERSION
    ):
        raise ValueError("unsupported student training run manifest")

    artifacts = _require_mapping(run_manifest.get("artifacts"), field="artifacts")
    checkpoint_entry = _require_mapping(
        artifacts.get("best_checkpoint"), field="artifacts.best_checkpoint"
    )
    checkpoint_path = _contained_artifact(
        run_manifest_path.parent,
        checkpoint_entry.get("relative_path"),
        field="best checkpoint path",
    )
    expected_checkpoint_sha256 = _require_sha256(
        checkpoint_entry.get("sha256"), field="best checkpoint SHA-256"
    )
    observed_checkpoint_sha256 = sha256_file(checkpoint_path)
    if observed_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("best checkpoint SHA-256 differs from the run manifest")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("student checkpoint must be an object")
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("format_version") != STUDENT_CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError("unsupported student checkpoint")

    dataset = _require_mapping(run_manifest.get("dataset"), field="dataset")
    dataset_manifest_sha256 = _require_sha256(
        dataset.get("manifest_sha256"), field="dataset manifest SHA-256"
    )
    if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("checkpoint and run manifest refer to different datasets")

    model_config = _model_config(checkpoint.get("model_config"))
    training_config = _training_config(checkpoint.get("training_config"))
    if model_config.landmark_indices != EXPECTED_LANDMARK_INDICES:
        raise ValueError(
            f"iOS MVP requires landmark indices {EXPECTED_LANDMARK_INDICES}, "
            f"got {model_config.landmark_indices}"
        )
    if training_config.image_size != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"iOS MVP requires {EXPECTED_IMAGE_SIZE}x{EXPECTED_IMAGE_SIZE} input, "
            f"got {training_config.image_size}"
        )

    model = FingertipDepthStudent(model_config, initial_depth_bias_m=0.0)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint.model_state_dict must be an object")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return StudentExportSource(
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=sha256_file(run_manifest_path),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=observed_checkpoint_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        checkpoint=checkpoint,
        model=model,
        model_config=model_config,
        training_config=training_config,
    )


# Core ML変換用に、固定バッチ・画像正規化を行うトレースラッパーを作ります。
def make_coreml_wrapper(source: StudentExportSource) -> CoreMLTraceWrapper:
    return CoreMLTraceWrapper(
        source.model,
        image_mean=source.training_config.image_mean,
        image_std=source.training_config.image_std,
    ).eval()


# torch.export用に、学習済みモデルと学習時の画像正規化設定を包むラッパーを作ります。
def make_export_wrapper(source: StudentExportSource) -> StudentCoreMLWrapper:
    """Build the unmodified deployment wrapper used by ``torch.export``."""

    return StudentCoreMLWrapper(
        source.model,
        image_mean=source.training_config.image_mean,
        image_std=source.training_config.image_std,
    ).eval()


# ExportedProgramのグラフに含まれる関数演算子を数え、名前順の辞書で返します。
def _torch_export_operator_counts(
    exported: torch.export.ExportedProgram,
) -> dict[str, int]:
    counts = Counter(
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    )
    return dict(sorted(counts.items()))


# グラフ形式がATENであることと未対応の融合演算子が残っていないことを確認し、監査情報を返します。
def _audit_torch_export_graph(
    exported: torch.export.ExportedProgram,
) -> dict[str, Any]:
    dialect = str(exported.dialect)
    if dialect != "ATEN":
        raise ValueError(f"Core ML requires an ATEN ExportedProgram, got {dialect}")
    operator_counts = _torch_export_operator_counts(exported)
    forbidden = sorted(
        operator
        for operator in operator_counts
        if operator.startswith(FORBIDDEN_TORCH_EXPORT_OPERATOR_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"unsupported fused operators remain after decomposition: {forbidden}")
    return {
        "dialect": dialect,
        "operator_counts": operator_counts,
        "forbidden_operators": forbidden,
        "node_count": len(tuple(exported.graph.nodes)),
    }


# 固定入力で厳密なtorch.exportグラフを生成し、元のPyTorch出力との誤差を確認して保存します。
def export_student_for_coreml(
    wrapper: nn.Module,
    *,
    output_path: Path,
    seed: int = 20260925,
) -> tuple[torch.export.ExportedProgram, dict[str, Any]]:
    """Capture a strict fixed-shape ATEN graph and verify it against eager PyTorch."""

    output_path = output_path.resolve()
    if output_path.suffix != ".pt2":
        raise ValueError("torch.export output must use the .pt2 suffix")
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing ExportedProgram: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    example_image = torch.rand(
        (1, 3, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE), generator=generator
    )
    example_landmarks = torch.rand(
        (1, len(EXPECTED_LANDMARK_INDICES), 2), generator=generator
    )
    verification_image = torch.rand(
        (1, 3, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE), generator=generator
    )
    verification_landmarks = torch.rand(
        (1, len(EXPECTED_LANDMARK_INDICES), 2), generator=generator
    )

    exported = torch.export.export(
        wrapper,
        (example_image, example_landmarks),
        strict=True,
    ).run_decompositions({})
    graph_audit = _audit_torch_export_graph(exported)
    with torch.inference_mode():
        eager_output = wrapper(verification_image, verification_landmarks)
        exported_output = exported.module()(verification_image, verification_landmarks)
    difference = (eager_output - exported_output).abs()
    maximum_difference_m = float(difference.max().item())
    if maximum_difference_m > 1e-6:
        raise ValueError(
            f"ExportedProgram differs from trained PyTorch by {maximum_difference_m:.9f} m"
        )
    torch.export.save(exported, output_path)
    return exported, {
        "case_count": 1,
        "trained_pytorch_vs_exported_program_mean_absolute_difference_m": float(
            difference.mean().item()
        ),
        "trained_pytorch_vs_exported_program_max_absolute_difference_m": maximum_difference_m,
        "fixed_shape_export": True,
        "batch_size": 1,
        "strict": True,
        "decomposition_table": "empty",
        **graph_audit,
    }


# 固定形状の推論ラッパーをTorchScriptに変換し、別入力でも元モデルと一致するか検証します。
def trace_student_for_coreml(
    wrapper: nn.Module,
    *,
    output_path: Path,
    seed: int = 20260925,
) -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
    """Trace the fixed-shape deployment wrapper and verify a second input."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    example_image = torch.rand(
        (1, 3, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE), generator=generator
    )
    example_landmarks = torch.rand((1, len(EXPECTED_LANDMARK_INDICES), 2), generator=generator)
    verification_image = torch.rand(
        (1, 3, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE), generator=generator
    )
    verification_landmarks = torch.rand((1, len(EXPECTED_LANDMARK_INDICES), 2), generator=generator)

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            (example_image, example_landmarks),
            check_trace=True,
            strict=True,
        )
        eager_output = wrapper(verification_image, verification_landmarks)
        traced_output = traced(verification_image, verification_landmarks)
        if isinstance(wrapper, CoreMLTraceWrapper):
            normalized_image = (verification_image - wrapper.image_mean) / wrapper.image_std
            centered_landmarks = verification_landmarks * 2.0 - 1.0
            pytorch_output = wrapper.model(normalized_image, centered_landmarks).unsqueeze(1)
        else:
            pytorch_output = eager_output
    difference = (eager_output - traced_output).abs()
    pytorch_difference = (eager_output - pytorch_output).abs()
    maximum_difference_m = float(difference.max().item())
    if maximum_difference_m > 1e-6:
        raise ValueError(
            f"TorchScript differs from its export wrapper by {maximum_difference_m:.9f} m"
        )
    pytorch_maximum_difference_m = float(pytorch_difference.max().item())
    if pytorch_maximum_difference_m > 1e-6:
        raise ValueError(
            "Core ML compatible operations differ from the trained PyTorch model by "
            f"{pytorch_maximum_difference_m:.9f} m"
        )
    traced.save(str(output_path))
    return traced, {
        "case_count": 1,
        "export_wrapper_vs_torchscript_mean_absolute_difference_m": float(difference.mean().item()),
        "export_wrapper_vs_torchscript_max_absolute_difference_m": maximum_difference_m,
        "trained_pytorch_vs_export_wrapper_mean_absolute_difference_m": float(
            pytorch_difference.mean().item()
        ),
        "trained_pytorch_vs_export_wrapper_max_absolute_difference_m": (
            pytorch_maximum_difference_m
        ),
        "fixed_shape_export": True,
        "batch_size": 1,
    }


# 指定されたPyTorchラッパーをML Program形式のCore MLモデルへ変換し、入出力名・対応OS・学習元情報を付けて保存します。
def _convert_to_coreml(
    source_model: Any,
    *,
    output_path: Path,
    checkpoint_sha256: str,
    export_backend: str,
    model_id: str | None,
    run_manifest_sha256: str | None,
) -> dict[str, Any]:
    try:
        import coremltools as ct
    except ImportError as error:  # pragma: no cover - exercised in the dedicated environment
        raise RuntimeError(
            "coremltools is required; use environments/coreml_export via uv"
        ) from error

    output_path = output_path.resolve()
    if output_path.suffix != ".mlpackage":
        raise ValueError("Core ML output must use the .mlpackage suffix")
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing Core ML package: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coreml_model = ct.convert(
        source_model,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.ImageType(
                name=IMAGE_INPUT_NAME,
                shape=(1, 3, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE),
                color_layout=ct.colorlayout.RGB,
                scale=1.0 / 255.0,
            ),
            ct.TensorType(
                name=LANDMARK_INPUT_NAME,
                shape=(1, len(EXPECTED_LANDMARK_INDICES), 2),
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name=OUTPUT_NAME, dtype=np.float32)],
    )
    coreml_model.author = "monocular-fingertip-depth"
    coreml_model.short_description = (
        "Phase 8 ViT + landmark Transformer: index fingertip optical-axis depth in metres"
    )
    coreml_model.version = "1"
    metadata = {
        "checkpoint_sha256": checkpoint_sha256,
        "input_landmark_indices": ",".join(map(str, EXPECTED_LANDMARK_INDICES)),
        "image_resize": "direct_bicubic_224x224_no_crop",
        "target_device": "iPhone 15 (iPhone15,4), iOS 26.6.1",
        "training_target": "Depth Pro pseudo-label metric Z in metres",
        "export_backend": export_backend,
    }
    if model_id is not None:
        metadata["model_id"] = model_id
    if run_manifest_sha256 is not None:
        metadata["run_manifest_sha256"] = run_manifest_sha256
    coreml_model.user_defined_metadata.update(metadata)
    coreml_model.save(str(output_path))

    spec = coreml_model.get_spec()
    return {
        "coremltools_version": str(ct.__version__),
        "model_type": str(spec.WhichOneof("Type")),
        "specification_version": int(spec.specificationVersion),
        "compute_precision": "float16",
        "minimum_deployment_target": "iOS 18 (application target is iOS 26.0)",
        "export_backend": export_backend,
        "model_id": model_id,
    }


# TorchScriptモデルを固定入出力のfloat16 Core MLモデルへ変換します。
def convert_torchscript_to_coreml(
    traced: torch.jit.ScriptModule,
    *,
    output_path: Path,
    checkpoint_sha256: str,
    model_id: str | None = None,
    run_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Convert the retained TorchScript fallback to a float16 ML Program."""

    return _convert_to_coreml(
        traced,
        output_path=output_path,
        checkpoint_sha256=checkpoint_sha256,
        export_backend=TORCHSCRIPT_BACKEND,
        model_id=model_id,
        run_manifest_sha256=run_manifest_sha256,
    )


# 監査済みATEN ExportedProgramをfloat16 Core MLモデルへ変換します。
def convert_exported_program_to_coreml(
    exported: torch.export.ExportedProgram,
    *,
    output_path: Path,
    checkpoint_sha256: str,
    model_id: str,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    """Convert an audited ATEN ExportedProgram to a float16 ML Program."""

    _audit_torch_export_graph(exported)
    return _convert_to_coreml(
        exported,
        output_path=output_path,
        checkpoint_sha256=checkpoint_sha256,
        export_backend=TORCH_EXPORT_BACKEND,
        model_id=model_id,
        run_manifest_sha256=run_manifest_sha256,
    )


# 軌跡デモの入力検証処理を使い、Core MLとの比較に使う実動画観測を読み込みます。
def load_parity_observations(run_manifest_path: Path) -> tuple[Any, ...]:
    """Load the exact identity observations used by the existing 2030 demo."""

    # デプロイ用ラッパーがデモ実装に依存しないよう、ここで遅延インポートします。
    from .student_trajectory_demo import load_demo_inputs

    return tuple(load_demo_inputs(run_manifest_path).observations)


# 実動画観測からRGB画像・ランドマーク・PyTorch予測をまとめた再現可能な比較用NPZを作ります。
def build_parity_fixture(
    wrapper: nn.Module,
    observations: Sequence[Any],
    *,
    output_path: Path,
    batch_size: int = 16,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write self-contained RGB/landmark/PyTorch vectors for macOS verification."""

    if not observations:
        raise ValueError("at least one parity observation is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    frame_indices: list[int] = []
    sample_ids: list[str] = []
    predictions: list[float] = []

    for start in range(0, len(observations), batch_size):
        rows = observations[start : start + batch_size]
        batch_images: list[np.ndarray] = []
        batch_landmarks: list[np.ndarray] = []
        for observation in rows:
            sample = observation.sample
            bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (sample.height, sample.width, 3):
                raise ValueError(f"cannot load parity image: {sample.image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb,
                (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE),
                interpolation=cv2.INTER_CUBIC,
            )
            landmark_xy = np.asarray(sample.landmark_xy, dtype=np.float32)
            if landmark_xy.shape != (len(EXPECTED_LANDMARK_INDICES), 2):
                raise ValueError(f"unexpected landmark shape: {sample.sample_id}")
            images.append(np.ascontiguousarray(resized, dtype=np.uint8))
            landmarks.append(landmark_xy)
            frame_indices.append(int(sample.frame_index))
            sample_ids.append(str(sample.sample_id))
            batch_images.append(resized)
            batch_landmarks.append(landmark_xy)

        image_tensor = torch.from_numpy(np.stack(batch_images)).permute(0, 3, 1, 2)
        image_tensor = image_tensor.to(dtype=torch.float32).div(255.0)
        landmark_tensor = torch.from_numpy(np.stack(batch_landmarks))
        with torch.inference_mode():
            output = wrapper(image_tensor, landmark_tensor).squeeze(1)
        predictions.extend(float(value) for value in output.cpu().tolist())
        if progress is not None:
            progress(
                f"parity fixture {min(start + batch_size, len(observations))}/{len(observations)}"
            )

    max_sample_id_length = max(len(value) for value in sample_ids)
    np.savez_compressed(
        output_path,
        images_rgb_uint8=np.stack(images),
        landmarks_xy=np.stack(landmarks),
        pytorch_depth_m=np.asarray(predictions, dtype=np.float32),
        frame_index=np.asarray(frame_indices, dtype=np.int32),
        sample_id=np.asarray(sample_ids, dtype=f"<U{max_sample_id_length}"),
    )
    return {
        "sample_count": len(observations),
        "sequence_id": "finger_movement_2030",
        "image_shape": [len(observations), EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE, 3],
        "landmark_shape": [len(observations), len(EXPECTED_LANDMARK_INDICES), 2],
        "prediction_min_m": float(np.min(predictions)),
        "prediction_max_m": float(np.max(predictions)),
    }


# 既存比較用NPZの画像とランドマークを再利用し、現在のPyTorchモデルの予測を再計算します。
def rebuild_parity_fixture(
    wrapper: nn.Module,
    source_fixture_path: Path,
    *,
    output_path: Path,
    batch_size: int = 16,
    limit: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Reuse audited RGB/landmark inputs while recomputing PyTorch predictions."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    source_fixture_path = source_fixture_path.resolve()
    if not source_fixture_path.is_file():
        raise FileNotFoundError(source_fixture_path)
    source_fixture_sha256 = sha256_file(source_fixture_path)
    with np.load(source_fixture_path, allow_pickle=False) as source_fixture:
        images = np.array(source_fixture["images_rgb_uint8"], copy=True)
        landmarks = np.array(source_fixture["landmarks_xy"], copy=True)
        frame_indices = np.array(source_fixture["frame_index"], copy=True)
        sample_ids = (
            np.array(source_fixture["sample_id"], copy=True)
            if "sample_id" in source_fixture.files
            else np.asarray(
                [f"frame:{int(frame_index)}" for frame_index in frame_indices],
                dtype=str,
            )
        )

    count = images.shape[0]
    if images.dtype != np.uint8 or images.shape != (
        count,
        EXPECTED_IMAGE_SIZE,
        EXPECTED_IMAGE_SIZE,
        3,
    ):
        raise ValueError("source parity images must be uint8 [N,224,224,3]")
    if landmarks.dtype != np.float32 or landmarks.shape != (
        count,
        len(EXPECTED_LANDMARK_INDICES),
        2,
    ):
        raise ValueError("source parity landmarks must be float32 [N,4,2]")
    if frame_indices.shape != (count,) or sample_ids.shape != (count,):
        raise ValueError("source parity arrays have inconsistent lengths")
    if count == 0:
        raise ValueError("source parity fixture must not be empty")
    if not np.all(np.isfinite(landmarks)) or np.any((landmarks < 0.0) | (landmarks > 1.0)):
        raise ValueError("source parity landmarks must be finite and in [0,1]")
    if limit:
        selected = slice(0, min(limit, count))
        images = images[selected]
        landmarks = landmarks[selected]
        frame_indices = frame_indices[selected]
        sample_ids = sample_ids[selected]
        count = images.shape[0]

    predictions: list[float] = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        image_tensor = torch.from_numpy(images[start:end]).permute(0, 3, 1, 2)
        image_tensor = image_tensor.to(dtype=torch.float32).div(255.0)
        landmark_tensor = torch.from_numpy(landmarks[start:end])
        with torch.inference_mode():
            output = wrapper(image_tensor, landmark_tensor).squeeze(1)
        predictions.extend(float(value) for value in output.cpu().tolist())
        if progress is not None:
            progress(f"parity fixture {end}/{count}")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        images_rgb_uint8=images,
        landmarks_xy=landmarks,
        pytorch_depth_m=np.asarray(predictions, dtype=np.float32),
        frame_index=frame_indices,
        sample_id=sample_ids,
    )
    return {
        "sample_count": count,
        "sequence_id": "finger_movement_2030",
        "image_shape": list(images.shape),
        "landmark_shape": list(landmarks.shape),
        "prediction_min_m": float(np.min(predictions)),
        "prediction_max_m": float(np.max(predictions)),
        "input_source_fixture_path": str(source_fixture_path),
        "input_source_fixture_sha256": source_fixture_sha256,
    }


# ファイルまたはディレクトリ内の全ファイルをハッシュし、サイズとSHA-256を記録します。
def artifact_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_file():
        byte_count = path.stat().st_size
        digest = sha256_file(path)
    elif path.is_dir():
        digest_builder = hashlib.sha256()
        byte_count = 0
        for child in sorted(value for value in path.rglob("*") if value.is_file()):
            relative = child.relative_to(path).as_posix().encode()
            child_digest = sha256_file(child).encode()
            digest_builder.update(len(relative).to_bytes(8, "big"))
            digest_builder.update(relative)
            digest_builder.update(child_digest)
            byte_count += child.stat().st_size
        digest = digest_builder.hexdigest()
    else:
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": digest, "byte_count": byte_count}


# 書出し・マニフェストを書き込みます。
def write_export_manifest(
    *,
    output_path: Path,
    source: StudentExportSource,
    torchscript_path: Path | None,
    coreml_path: Path,
    parity_fixture_path: Path,
    trace_validation: Mapping[str, Any] | None,
    coreml_conversion: Mapping[str, Any],
    parity_fixture: Mapping[str, Any],
    exported_program_path: Path | None = None,
    export_validation: Mapping[str, Any] | None = None,
    model_id: str = "phase8_student",
    export_backend: str | None = None,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_backend = export_backend or (
        TORCH_EXPORT_BACKEND if exported_program_path is not None else TORCHSCRIPT_BACKEND
    )
    if selected_backend == TORCH_EXPORT_BACKEND:
        if exported_program_path is None or export_validation is None:
            raise ValueError("torch.export manifest requires its .pt2 artifact and validation")
        graph_artifacts = {"exported_program": artifact_record(exported_program_path)}
        graph_validation = {"torch_export": dict(export_validation)}
        transformer_strategy = (
            "strict torch.export followed by run_decompositions({}); ATEN graph audited"
        )
    elif selected_backend == TORCHSCRIPT_BACKEND:
        if torchscript_path is None or trace_validation is None:
            raise ValueError("TorchScript manifest requires its trace artifact and validation")
        graph_artifacts = {"torchscript": artifact_record(torchscript_path)}
        graph_validation = {"torchscript": dict(trace_validation)}
        transformer_strategy = (
            "evaluation-equivalent explicit QKV matmul/softmax; trained weights unchanged"
        )
    else:
        raise ValueError(f"unsupported export backend: {selected_backend}")
    manifest: dict[str, Any] = {
        "format": COREML_EXPORT_FORMAT,
        "format_version": COREML_EXPORT_FORMAT_VERSION,
        "model": {
            "model_id": model_id,
            "export_backend": selected_backend,
        },
        "target": {
            "device": TARGET_DEVICE,
            "camera": {
                "position": "back",
                "device_type": "builtInWideAngleCamera",
                "resolution": [1080, 1920],
                "frames_per_second": 30,
                "orientation": "portrait",
                "full_frame_equivalent_focal_length_mm": 36.0,
                "native_equivalent_focal_length_mm": 26.0,
                "requested_zoom_factor": 36.0 / 26.0,
                "stabilization": "off",
            },
        },
        "source": {
            "run_manifest_path": str(source.run_manifest_path),
            "run_manifest_sha256": source.run_manifest_sha256,
            "checkpoint_path": str(source.checkpoint_path),
            "checkpoint_sha256": source.checkpoint_sha256,
            "checkpoint_epoch": int(source.checkpoint.get("epoch", -1)),
            "dataset_manifest_sha256": source.dataset_manifest_sha256,
        },
        "interface": {
            "image": {
                "name": IMAGE_INPUT_NAME,
                "coreml_type": "RGB image",
                "size": [EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE],
                "source_range": [0, 255],
                "coreml_scale": 1.0 / 255.0,
                "resize": "whole portrait frame directly to 224x224; no crop or letterbox",
                "normalization_mean": list(source.training_config.image_mean),
                "normalization_std": list(source.training_config.image_std),
            },
            "landmarks": {
                "name": LANDMARK_INPUT_NAME,
                "shape": [1, len(EXPECTED_LANDMARK_INDICES), 2],
                "indices": list(EXPECTED_LANDMARK_INDICES),
                "coordinate_range": [0.0, 1.0],
                "internal_transform": "2*x-1",
                "relative_z_included": False,
            },
            "output": {
                "name": OUTPUT_NAME,
                "shape": [1, 1],
                "unit": "metre",
                "meaning": "index fingertip camera-axis Z; raw scalar without clipping",
            },
        },
        "conversion": {
            **dict(coreml_conversion),
            "export_backend": selected_backend,
            "transformer_export_strategy": transformer_strategy,
        },
        "validation": {
            **graph_validation,
            "parity_fixture": dict(parity_fixture),
            "coreml_runtime": {
                "status": "pending_macos_or_ios",
                "mean_absolute_difference_limit_m": 0.001,
                "max_absolute_difference_limit_m": 0.003,
                "reason": "Core ML prediction is unavailable on Linux",
            },
        },
        "artifacts": {
            **graph_artifacts,
            "coreml": artifact_record(coreml_path),
            "parity_fixture": artifact_record(parity_fixture_path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
        },
    }
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return manifest


__all__ = [
    "COREML_EXPORT_FORMAT",
    "COREML_EXPORT_FORMAT_VERSION",
    "EXPECTED_IMAGE_SIZE",
    "EXPECTED_LANDMARK_INDICES",
    "FORBIDDEN_TORCH_EXPORT_OPERATOR_PREFIXES",
    "IMAGE_INPUT_NAME",
    "LANDMARK_INPUT_NAME",
    "OUTPUT_NAME",
    "TORCHSCRIPT_BACKEND",
    "TORCH_EXPORT_BACKEND",
    "CoreMLTraceWrapper",
    "StudentCoreMLWrapper",
    "StudentExportSource",
    "artifact_record",
    "build_parity_fixture",
    "convert_exported_program_to_coreml",
    "convert_torchscript_to_coreml",
    "export_student_for_coreml",
    "load_parity_observations",
    "load_student_export_source",
    "make_coreml_wrapper",
    "make_export_wrapper",
    "rebuild_parity_fixture",
    "sha256_file",
    "trace_student_for_coreml",
    "write_export_manifest",
]
