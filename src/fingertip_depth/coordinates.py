"""Coordinate conversion and depth lookup for MediaPipe landmarks."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class NormalizedLandmark(Protocol):
    x: float
    y: float


def normalized_to_pixel(x: float, y: float, *, width: int, height: int) -> tuple[int, int]:
    """Map normalized coordinates to an image pixel using the experiment-plan rule.

    The plan specifies ``round(x * W), round(y * H)``. Mathematical half-up
    rounding is used rather than Python's ties-to-even ``round``; clipping keeps
    the exact boundary value 1.0 inside the image.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("normalized coordinates must be finite")
    if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
        raise ValueError("normalized coordinates must be in [0, 1]")

    u = math.floor(x * width + 0.5)
    v = math.floor(y * height + 0.5)
    return min(max(u, 0), width - 1), min(max(v, 0), height - 1)


def lookup_depth(depth_m: np.ndarray, u: int, v: int) -> float:
    """Return D(v, u) in metres after strict bounds and validity checks."""

    if depth_m.ndim != 2:
        raise ValueError("depth map must have shape (height, width)")
    height, width = depth_m.shape
    if not (0 <= u < width and 0 <= v < height):
        raise IndexError(f"pixel ({u}, {v}) is outside depth map {width}x{height}")
    value = float(depth_m[v, u])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"depth at ({u}, {v}) is not a finite positive value")
    return value
