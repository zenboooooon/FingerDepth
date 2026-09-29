'ピンホールカメラの内部パラメーターと光軸方向の深度から、カメラ座標系の三次元位置を計算します。座標軸は右向きX、下向きY、前向きZです。\n\n深度はカメラ光軸方向のZ距離（メートル）として扱い、座標軸は右向きX・下向きY・前向きZです。'

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .camera import CameraIntrinsics

CAMERA_COORDINATE_CONVENTION = "x-right, y-down, z-forward"


# カメラ座標系の三次元位置と、対応する画像上の画素位置を保持します。
@dataclass(frozen=True, slots=True)
class CameraPoint3D:
    """A finite metric point in the camera coordinate system."""

    x_m: float
    y_m: float
    z_m: float

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x_m, self.y_m, self.z_m)):
            raise ValueError("camera point coordinates must be finite")
        if self.z_m <= 0.0:
            raise ValueError("camera point z_m must be positive")

    # 主要なフィールドを、JSONへ保存できる辞書に変換します。
    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# 画素座標と光軸深度をカメラ座標系の三次元位置へ逆投影します。
def backproject_pixel(
    *,
    u_px: float,
    v_px: float,
    z_m: float,
    intrinsics: CameraIntrinsics,
) -> CameraPoint3D:
    """Back-project one pixel and optical-axis depth into camera coordinates."""

    if not all(math.isfinite(value) for value in (u_px, v_px, z_m)):
        raise ValueError("pixel coordinates and depth must be finite")
    if z_m <= 0.0:
        raise ValueError("z_m must be positive")
    return CameraPoint3D(
        x_m=(float(u_px) - intrinsics.cx_px) * float(z_m) / intrinsics.fx_px,
        y_m=(float(v_px) - intrinsics.cy_px) * float(z_m) / intrinsics.fy_px,
        z_m=float(z_m),
    )


# カメラ座標系の三次元位置を画像上の画素座標へ投影します。
def project_camera_point(
    point: CameraPoint3D,
    *,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Project a camera-coordinate point into pixel coordinates."""

    return (
        intrinsics.fx_px * point.x_m / point.z_m + intrinsics.cx_px,
        intrinsics.fy_px * point.y_m / point.z_m + intrinsics.cy_px,
    )


__all__ = [
    "CAMERA_COORDINATE_CONVENTION",
    "CameraPoint3D",
    "backproject_pixel",
    "project_camera_point",
]
