import pytest

from fingertip_depth.camera import CameraIntrinsics


def test_centered_intrinsics() -> None:
    camera = CameraIntrinsics.centered(width=640, height=480, fx_px=500.0)
    assert camera.fx_px == 500.0
    assert camera.fy_px == 500.0
    assert camera.cx_px == 319.5
    assert camera.cy_px == 239.5


def test_scaled_intrinsics() -> None:
    camera = CameraIntrinsics(500.0, 510.0, 320.0, 240.0).scaled(0.5)
    assert camera == CameraIntrinsics(250.0, 255.0, 160.0, 120.0)


@pytest.mark.parametrize("fx,fy", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_rejects_invalid_focal_length(fx: float, fy: float) -> None:
    with pytest.raises(ValueError):
        CameraIntrinsics(fx, fy, 0.0, 0.0)
