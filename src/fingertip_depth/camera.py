"""Camera-intrinsic handling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels for the original RGB frame."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self) -> None:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("fx_px and fy_px must be positive")

    @classmethod
    def centered(
        cls,
        *,
        width: int,
        height: int,
        fx_px: float,
        fy_px: float | None = None,
    ) -> CameraIntrinsics:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return cls(
            fx_px=float(fx_px),
            fy_px=float(fx_px if fy_px is None else fy_px),
            cx_px=(width - 1) / 2.0,
            cy_px=(height - 1) / 2.0,
        )

    def scaled(self, scale: float) -> CameraIntrinsics:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        return CameraIntrinsics(
            fx_px=self.fx_px * scale,
            fy_px=self.fy_px * scale,
            cx_px=self.cx_px * scale,
            cy_px=self.cy_px * scale,
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)
