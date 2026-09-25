import numpy as np
import torch

from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.constants import METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH
from fingertip_depth.metric3d import prepare_metric3d_input, restore_metric_depth


def test_restore_removes_odd_horizontal_padding_from_portrait_input() -> None:
    rgb = np.zeros((960, 641, 3), dtype=np.uint8)
    _tensor, geometry = prepare_metric3d_input(rgb)
    assert geometry.pad_top == geometry.pad_bottom == 0
    assert geometry.pad_right == geometry.pad_left + 1

    canonical = torch.full(
        (1, 1, METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH),
        99.0,
        dtype=torch.float32,
    )
    row_end = METRIC3D_INPUT_HEIGHT - geometry.pad_bottom
    column_end = METRIC3D_INPUT_WIDTH - geometry.pad_right
    canonical[
        :,
        :,
        geometry.pad_top : row_end,
        geometry.pad_left : column_end,
    ] = 2.5
    camera = CameraIntrinsics(
        1000.0 / geometry.scale,
        1000.0 / geometry.scale,
        320.0,
        479.5,
    )

    restored = restore_metric_depth(
        canonical,
        geometry=geometry,
        original_intrinsics=camera,
    )

    assert restored.shape == (960, 641)
    assert np.allclose(restored, 2.5, atol=1e-6)
