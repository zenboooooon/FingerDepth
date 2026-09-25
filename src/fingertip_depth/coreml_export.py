"""Audited Core ML export for the Phase 8 fingertip-depth student."""

from __future__ import annotations

import hashlib
import json
import platform
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
COREML_EXPORT_FORMAT_VERSION = 1
CHECKPOINT_FORMAT = "fingertip-depth-student-checkpoint"
TRAINING_RUN_FORMAT = "fingertip-depth-student-training"
TRAINING_RUN_FORMAT_VERSION = 2
EXPECTED_LANDMARK_INDICES = (5, 6, 7, 8)
EXPECTED_IMAGE_SIZE = 224
IMAGE_INPUT_NAME = "image"
LANDMARK_INPUT_NAME = "landmarks_xy"
OUTPUT_NAME = "depth_m"
TARGET_DEVICE = {
    "marketing_name": "iPhone 15",
    "hardware_identifier": "iPhone15,4",
    "operating_system": "iOS 26.6.1",
}


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


class StudentCoreMLWrapper(nn.Module):
    """Expose deployment-friendly raw RGB and normalized landmark inputs.

    Core ML's image input scales uint8 RGB pixels by ``1 / 255``. Therefore
    ``image_rgb_0_1`` is already in [0, 1] when this module is executed. The
    wrapper owns the remaining ImageNet normalization and the landmark
    centering so the Swift application cannot accidentally omit or duplicate
    either operation.
    """

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

    def forward(
        self,
        image_rgb_0_1: torch.Tensor,
        landmarks_xy_0_1: torch.Tensor,
    ) -> torch.Tensor:
        normalized_image = (image_rgb_0_1 - self.image_mean) / self.image_std
        centered_landmarks = landmarks_xy_0_1 * 2.0 - 1.0
        # A [batch, 1] output produces one stable Core ML MultiArray feature.
        return self.model(normalized_image, centered_landmarks).unsqueeze(1)


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

    def _encode_image_fixed_batch(self, image: torch.Tensor) -> torch.Tensor:
        encoder = self.model.image_encoder
        tokens = encoder.patch_embed(image)
        # cls_token already has batch dimension 1. Avoiding expand() removes a
        # dynamic tensor-to-int conversion from the traced graph.
        tokens = torch.cat((encoder.cls_token, tokens), dim=1)
        tokens = encoder.pos_drop(tokens + encoder.pos_embed)
        tokens = encoder.patch_drop(tokens)
        tokens = encoder.norm_pre(tokens)
        tokens = encoder.blocks(tokens)
        return encoder.norm(tokens)

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    return digest


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


def _model_config(raw: object) -> StudentModelConfig:
    config = _require_mapping(raw, field="checkpoint.model_config")
    allowed = {field.name for field in fields(StudentModelConfig)}
    values = {name: config[name] for name in allowed if name in config}
    # Loading a trained state dict must never trigger a network download.
    values["pretrained_image_encoder"] = False
    if "landmark_indices" in values:
        values["landmark_indices"] = tuple(int(value) for value in values["landmark_indices"])
    return StudentModelConfig(**values)


def _training_config(raw: object) -> StudentTrainingConfig:
    config = _require_mapping(raw, field="checkpoint.training_config")
    allowed = {field.name for field in fields(StudentTrainingConfig)}
    values = {name: config[name] for name in allowed if name in config}
    if "image_mean" in values:
        values["image_mean"] = tuple(float(value) for value in values["image_mean"])
    if "image_std" in values:
        values["image_std"] = tuple(float(value) for value in values["image_std"])
    return StudentTrainingConfig(**values)


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


def make_coreml_wrapper(source: StudentExportSource) -> CoreMLTraceWrapper:
    return CoreMLTraceWrapper(
        source.model,
        image_mean=source.training_config.image_mean,
        image_std=source.training_config.image_std,
    ).eval()


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


def convert_torchscript_to_coreml(
    traced: torch.jit.ScriptModule,
    *,
    output_path: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Convert to a float16 ML Program with an RGB image input."""

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
        traced,
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
    coreml_model.user_defined_metadata.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "input_landmark_indices": ",".join(map(str, EXPECTED_LANDMARK_INDICES)),
            "image_resize": "direct_bicubic_224x224_no_crop",
            "target_device": "iPhone 15 (iPhone15,4), iOS 26.6.1",
            "training_target": "Depth Pro pseudo-label metric Z in metres",
        }
    )
    coreml_model.save(str(output_path))

    spec = coreml_model.get_spec()
    return {
        "coremltools_version": str(ct.__version__),
        "model_type": str(spec.WhichOneof("Type")),
        "specification_version": int(spec.specificationVersion),
        "compute_precision": "float16",
        "minimum_deployment_target": "iOS 18 (application target is iOS 26.0)",
    }


def load_parity_observations(run_manifest_path: Path) -> tuple[Any, ...]:
    """Load the exact identity observations used by the existing 2030 demo."""

    # Imported lazily to keep the deployment wrapper independent from demo code.
    from .student_trajectory_demo import load_demo_inputs

    return tuple(load_demo_inputs(run_manifest_path).observations)


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


def write_export_manifest(
    *,
    output_path: Path,
    source: StudentExportSource,
    torchscript_path: Path,
    coreml_path: Path,
    parity_fixture_path: Path,
    trace_validation: Mapping[str, Any],
    coreml_conversion: Mapping[str, Any],
    parity_fixture: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": COREML_EXPORT_FORMAT,
        "format_version": COREML_EXPORT_FORMAT_VERSION,
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
            "transformer_export_strategy": (
                "evaluation-equivalent explicit QKV matmul/softmax; trained weights unchanged"
            ),
        },
        "validation": {
            "torchscript": dict(trace_validation),
            "parity_fixture": dict(parity_fixture),
            "coreml_runtime": {
                "status": "pending_macos_or_ios",
                "mean_absolute_difference_limit_m": 0.001,
                "max_absolute_difference_limit_m": 0.003,
                "reason": "Core ML prediction is unavailable on Linux",
            },
        },
        "artifacts": {
            "torchscript": artifact_record(torchscript_path),
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
    "IMAGE_INPUT_NAME",
    "LANDMARK_INPUT_NAME",
    "OUTPUT_NAME",
    "CoreMLTraceWrapper",
    "StudentCoreMLWrapper",
    "StudentExportSource",
    "artifact_record",
    "build_parity_fixture",
    "convert_torchscript_to_coreml",
    "load_parity_observations",
    "load_student_export_source",
    "make_coreml_wrapper",
    "sha256_file",
    "trace_student_for_coreml",
    "write_export_manifest",
]
