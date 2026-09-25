import numpy as np
import pytest
import torch

from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.constants import METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH
from fingertip_depth.metric3d import prepare_metric3d_input, restore_metric_depth


def test_preprocess_preserves_ratio_and_nearly_zero_normalizes_padding() -> None:
    rgb = np.full((375, 1242, 3), 64, dtype=np.uint8)
    tensor, geometry = prepare_metric3d_input(rgb)
    assert tuple(tensor.shape) == (1, 3, METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH)
    assert geometry.resized_width == METRIC3D_INPUT_WIDTH
    assert geometry.resized_height == int(375 * (METRIC3D_INPUT_WIDTH / 1242))
    assert geometry.pad_top + geometry.pad_bottom + geometry.resized_height == 616
    # OpenCV rounds the official float padding values when the source is uint8.
    assert torch.all(torch.abs(tensor[0, :, 0, 0]) < 0.01)


def test_restore_unpads_resizes_and_decanonicalizes() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    _tensor, geometry = prepare_metric3d_input(rgb)
    canonical = torch.ones((1, 1, 616, 1064), dtype=torch.float32)
    camera = CameraIntrinsics(800.0, 800.0, 99.5, 49.5)
    restored = restore_metric_depth(
        canonical,
        geometry=geometry,
        original_intrinsics=camera,
    )
    expected = camera.fx_px * geometry.scale / 1000.0
    assert restored.shape == (100, 200)
    assert restored.dtype == np.float32
    assert np.allclose(restored, expected, atol=1e-6)

    doubled_focal = restore_metric_depth(
        canonical,
        geometry=geometry,
        original_intrinsics=CameraIntrinsics(1600.0, 1600.0, 99.5, 49.5),
    )
    assert np.allclose(doubled_focal, restored * 2.0, atol=1e-6)


def test_preprocess_rejects_bgr_like_shape_or_float() -> None:
    with pytest.raises(ValueError):
        prepare_metric3d_input(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        prepare_metric3d_input(np.zeros((10, 10, 3), dtype=np.float32))
