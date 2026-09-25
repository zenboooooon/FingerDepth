import math

import pytest

from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.geometry import CameraPoint3D, backproject_pixel, project_camera_point


def test_backproject_principal_point_is_on_optical_axis() -> None:
    intrinsics = CameraIntrinsics(500.0, 600.0, 319.5, 239.5)

    point = backproject_pixel(
        u_px=intrinsics.cx_px,
        v_px=intrinsics.cy_px,
        z_m=0.25,
        intrinsics=intrinsics,
    )

    assert point == CameraPoint3D(x_m=0.0, y_m=0.0, z_m=0.25)


def test_backproject_uses_x_right_y_down_z_forward_and_round_trips() -> None:
    intrinsics = CameraIntrinsics(100.0, 200.0, 10.0, 20.0)

    point = backproject_pixel(u_px=110.0, v_px=220.0, z_m=2.0, intrinsics=intrinsics)

    assert point == CameraPoint3D(x_m=2.0, y_m=2.0, z_m=2.0)
    assert project_camera_point(point, intrinsics=intrinsics) == pytest.approx((110.0, 220.0))
    assert point.as_dict() == {"x_m": 2.0, "y_m": 2.0, "z_m": 2.0}


@pytest.mark.parametrize(
    ("u_px", "v_px", "z_m"),
    [
        (math.nan, 0.0, 1.0),
        (0.0, math.inf, 1.0),
        (0.0, 0.0, math.nan),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ],
)
def test_backproject_rejects_invalid_values(u_px: float, v_px: float, z_m: float) -> None:
    intrinsics = CameraIntrinsics.centered(width=10, height=10, fx_px=100.0)
    with pytest.raises(ValueError):
        backproject_pixel(u_px=u_px, v_px=v_px, z_m=z_m, intrinsics=intrinsics)


@pytest.mark.parametrize(
    "point",
    [
        (math.nan, 0.0, 1.0),
        (0.0, math.inf, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ],
)
def test_camera_point_rejects_invalid_coordinates(point: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        CameraPoint3D(*point)
