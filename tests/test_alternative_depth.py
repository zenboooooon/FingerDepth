from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fingertip_depth import alternative_depth
from fingertip_depth.alternative_depth import (
    DepthProEstimator,
    UniDepthV2L,
    ensure_unidepth_checkpoint,
    verify_checkpoint,
)
from fingertip_depth.camera import CameraIntrinsics


class FakeUniDepth:
    def __init__(self) -> None:
        self.rgb: torch.Tensor | None = None
        self.camera: torch.Tensor | None = None

    def infer(self, rgb: torch.Tensor, camera: torch.Tensor | None) -> dict[str, torch.Tensor]:
        self.rgb = rgb
        self.camera = camera
        height, width = rgb.shape[-2:]
        return {
            "depth": torch.full((1, 1, height, width), 1.25),
            "intrinsics": torch.tensor(
                [[[800.0, 0.0, 3.5], [0.0, 801.0, 2.5], [0.0, 0.0, 1.0]]]
            ),
        }


class FakeDepthPro:
    def __init__(self) -> None:
        self.focal: torch.Tensor | None = None

    def infer(
        self, tensor: torch.Tensor, *, f_px: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        self.focal = f_px
        height, width = tensor.shape[-2:]
        focal = torch.tensor(900.0) if f_px is None else f_px
        return {
            "depth": torch.full((height, width), 2.5),
            "focallength_px": focal,
        }


def _rgb() -> np.ndarray:
    return np.zeros((6, 8, 3), dtype=np.uint8)


def _camera() -> CameraIntrinsics:
    return CameraIntrinsics.centered(width=8, height=6, fx_px=700.0, fy_px=710.0)


def test_unidepth_passes_original_rgb_and_k_without_extra_scaling() -> None:
    model = FakeUniDepth()
    estimator = UniDepthV2L(device="cpu", model=model)

    prediction = estimator.predict(_rgb(), intrinsics=_camera(), camera_mode="approx_k")

    assert model.rgb is not None
    assert tuple(model.rgb.shape) == (3, 6, 8)
    assert model.rgb.dtype == torch.uint8
    assert model.camera is not None
    assert torch.equal(
        model.camera,
        torch.tensor([[700.0, 0.0, 3.5], [0.0, 710.0, 2.5], [0.0, 0.0, 1.0]]),
    )
    assert prediction.depth_m.shape == (6, 8)
    assert prediction.depth_m.dtype == np.float32
    assert np.all(prediction.depth_m == 1.25)
    assert prediction.extras["predicted_intrinsics"]["fx_px"] == 800.0
    assert estimator.metadata["checkpoint_revision"] == alternative_depth.UNIDEPTH_HF_REVISION
    assert estimator.model_metadata == estimator.metadata


def test_unidepth_no_camera_passes_none() -> None:
    model = FakeUniDepth()
    estimator = UniDepthV2L(device="cpu", model=model)

    prediction = estimator.predict(_rgb(), intrinsics=None, camera_mode="no_camera")

    assert model.camera is None
    assert prediction.extras["input_intrinsics"] is None


@pytest.mark.parametrize(
    ("mode", "intrinsics"),
    [("approx_k", None), ("no_camera", _camera()), ("wrong", None)],
)
def test_unidepth_rejects_inconsistent_camera_modes(
    mode: str, intrinsics: CameraIntrinsics | None
) -> None:
    estimator = UniDepthV2L(device="cpu", model=FakeUniDepth())
    with pytest.raises(ValueError):
        estimator.predict(_rgb(), intrinsics=intrinsics, camera_mode=mode)


def test_depth_pro_passes_scalar_tensor_focal() -> None:
    model = FakeDepthPro()

    def transform(rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()

    estimator = DepthProEstimator(device="cpu", model=model, transform=transform)
    prediction = estimator.predict(
        _rgb(), intrinsics=_camera(), camera_mode="approx_focal"
    )

    assert isinstance(model.focal, torch.Tensor)
    assert model.focal.ndim == 0
    assert model.focal.dtype == torch.float32
    assert model.focal.item() == 700.0
    assert prediction.depth_m.shape == (6, 8)
    assert np.all(prediction.depth_m == 2.5)
    assert prediction.extras["input_focal_px"] == 700.0
    assert prediction.extras["output_focal_px"] == 700.0


def test_depth_pro_estimated_focal_passes_none() -> None:
    model = FakeDepthPro()
    estimator = DepthProEstimator(
        device="cpu",
        model=model,
        transform=lambda rgb: torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
    )

    prediction = estimator.predict(
        _rgb(), intrinsics=None, camera_mode="estimated_focal"
    )

    assert model.focal is None
    assert prediction.extras["input_focal_px"] is None
    assert prediction.extras["output_focal_px"] == 900.0


@pytest.mark.parametrize(
    ("rgb", "error"),
    [
        (np.zeros((6, 8), dtype=np.uint8), ValueError),
        (np.zeros((6, 8, 3), dtype=np.float32), ValueError),
        ("not an array", TypeError),
    ],
)
def test_adapters_validate_rgb(rgb: Any, error: type[Exception]) -> None:
    estimator = UniDepthV2L(device="cpu", model=FakeUniDepth())
    with pytest.raises(error):
        estimator.predict(rgb, intrinsics=None, camera_mode="no_camera")


def test_verified_download_uses_pinned_hugging_face_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixed test checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(alternative_depth, "UNIDEPTH_CHECKPOINT_SHA256", digest)
    received: dict[str, Any] = {}

    def download(**kwargs: Any) -> str:
        received.update(kwargs)
        return str(checkpoint)

    assert ensure_unidepth_checkpoint(downloader=download) == checkpoint
    assert received == {
        "repo_id": alternative_depth.UNIDEPTH_HF_REPOSITORY,
        "revision": alternative_depth.UNIDEPTH_HF_REVISION,
        "filename": alternative_depth.UNIDEPTH_CHECKPOINT_FILENAME,
        "repo_type": "model",
    }


def test_checkpoint_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_checkpoint(
            checkpoint,
            expected_sha256="0" * 64,
            model_name="test model",
        )
