"""Pinned Metric3D v2 inference adapter."""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from .camera import CameraIntrinsics
from .constants import (
    METRIC3D_CANONICAL_FOCAL_PX,
    METRIC3D_CHECKPOINT_URL,
    METRIC3D_HUB_MODEL,
    METRIC3D_HUB_REPO,
    METRIC3D_INPUT_HEIGHT,
    METRIC3D_INPUT_WIDTH,
    METRIC3D_MAX_DEPTH_M,
)

_RGB_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_RGB_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
_CHECKPOINT_FILENAME = "metric_depth_vit_small_800k.pth"
_CHECKPOINT_SHA256 = "b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6"


@dataclass(frozen=True, slots=True)
class PreprocessGeometry:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    scale: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


@dataclass(frozen=True, slots=True)
class DepthPrediction:
    depth_m: np.ndarray
    inference_ms: float
    device: str


def prepare_metric3d_input(rgb: np.ndarray) -> tuple[torch.Tensor, PreprocessGeometry]:
    """Apply the exact resize, mean padding and normalization from Metric3D hubconf."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB input must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        raise ValueError("RGB input must use uint8 values")

    original_height, original_width = rgb.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError("RGB input must not be empty")

    scale = min(
        METRIC3D_INPUT_HEIGHT / original_height,
        METRIC3D_INPUT_WIDTH / original_width,
    )
    resized_height = max(1, int(original_height * scale))
    resized_width = max(1, int(original_width * scale))
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_height = METRIC3D_INPUT_HEIGHT - resized_height
    pad_width = METRIC3D_INPUT_WIDTH - resized_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=_RGB_MEAN.tolist(),
    )

    normalized = (padded.astype(np.float32) - _RGB_MEAN) / _RGB_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    geometry = PreprocessGeometry(
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
        scale=scale,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )
    return tensor, geometry


def restore_metric_depth(
    canonical_depth: torch.Tensor,
    *,
    geometry: PreprocessGeometry,
    original_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Undo padding/resize and de-canonicalize Metric3D depth into metres."""

    depth = canonical_depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Metric3D returned unexpected depth shape {tuple(canonical_depth.shape)}")

    row_end = depth.shape[0] - geometry.pad_bottom if geometry.pad_bottom else depth.shape[0]
    col_end = depth.shape[1] - geometry.pad_right if geometry.pad_right else depth.shape[1]
    depth = depth[
        geometry.pad_top : row_end,
        geometry.pad_left : col_end,
    ]
    if tuple(depth.shape) != (geometry.resized_height, geometry.resized_width):
        raise ValueError(
            "unpadded Metric3D output does not match the preprocessed image: "
            f"got {tuple(depth.shape)}, expected "
            f"{(geometry.resized_height, geometry.resized_width)}"
        )

    depth = torch_f.interpolate(
        depth[None, None],
        size=(geometry.original_height, geometry.original_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Metric3D predicts in a camera with a 1000 px canonical focal length.
    # Resizing the RGB frame also scales its effective focal length.
    scaled_focal_px = original_intrinsics.fx_px * geometry.scale
    depth = depth * (scaled_focal_px / METRIC3D_CANONICAL_FOCAL_PX)
    depth = torch.clamp(depth, min=0.0, max=METRIC3D_MAX_DEPTH_M)
    return depth.detach().to(device="cpu", dtype=torch.float32).numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> str:
    actual = _sha256(path)
    if actual != _CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Metric3D checkpoint SHA-256 mismatch: expected {_CHECKPOINT_SHA256}, got {actual}"
        )
    return actual


def ensure_cached_checkpoint() -> Path:
    """Download raw bytes, verify them, and only then expose them to torch.load."""

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / _CHECKPOINT_FILENAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        _verify_checkpoint(checkpoint)
        return checkpoint

    with tempfile.NamedTemporaryFile(
        prefix=f"{_CHECKPOINT_FILENAME}.",
        suffix=".download",
        dir=checkpoint.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.hub.download_url_to_file(
            METRIC3D_CHECKPOINT_URL,
            str(temporary_path),
            progress=True,
        )
        _verify_checkpoint(temporary_path)
        temporary_path.replace(checkpoint)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint


def verify_cached_checkpoint() -> str:
    """Verify the currently cached fixed checkpoint without loading it."""

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / _CHECKPOINT_FILENAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Metric3D checkpoint cache was not created: {checkpoint}")
    return _verify_checkpoint(checkpoint)


class Metric3Dv2:
    """Metric3D v2 ViT-Small fixed at the experiment's phase-1 revision."""

    def __init__(self, *, device: str = "auto") -> None:
        self.device = self._resolve_device(device)
        self._model: Any | None = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("the pinned Metric3D v2 implementation requires a CUDA GPU")
            return torch.device("cuda")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("the pinned Metric3D v2 implementation only supports CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return resolved

    @property
    def model(self) -> Any:
        if self._model is None:
            checkpoint = ensure_cached_checkpoint()
            model = torch.hub.load(
                METRIC3D_HUB_REPO,
                METRIC3D_HUB_MODEL,
                pretrain=False,
                trust_repo=True,
            )
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(payload["model_state_dict"], strict=False)
            self._model = model.to(self.device).eval()
        return self._model

    def predict(self, rgb: np.ndarray, intrinsics: CameraIntrinsics) -> DepthPrediction:
        tensor, geometry = prepare_metric3d_input(rgb)
        tensor = tensor.to(self.device, non_blocking=True)
        model = self.model

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            canonical_depth, _confidence, _output = model.inference({"input": tensor})
        torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        depth_m = restore_metric_depth(
            canonical_depth,
            geometry=geometry,
            original_intrinsics=intrinsics,
        )
        return DepthPrediction(
            depth_m=depth_m,
            inference_ms=inference_ms,
            device=str(self.device),
        )
