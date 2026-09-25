import math

import numpy as np
import pytest

from fingertip_depth.constants import FINGERTIP_LANDMARK_INDEX
from fingertip_depth.coordinates import lookup_depth, normalized_to_pixel


def test_index_finger_tip_is_landmark_8() -> None:
    assert FINGERTIP_LANDMARK_INDEX == 8


@pytest.mark.parametrize(
    ("x", "y", "width", "height", "expected"),
    [
        (0.0, 0.0, 640, 480, (0, 0)),
        (1.0, 1.0, 640, 480, (639, 479)),
        (0.5, 0.5, 640, 480, (320, 240)),
        (0.25, 0.75, 10, 20, (3, 15)),
        (1.0, 1.0, 1, 1, (0, 0)),
    ],
)
def test_plan_rounding_and_clipping(
    x: float,
    y: float,
    width: int,
    height: int,
    expected: tuple[int, int],
) -> None:
    assert normalized_to_pixel(x, y, width=width, height=height) == expected


@pytest.mark.parametrize("value", [-0.001, 1.001, math.nan, math.inf])
def test_rejects_invalid_normalized_coordinate(value: float) -> None:
    with pytest.raises(ValueError):
        normalized_to_pixel(value, 0.5, width=10, height=10)


def test_depth_lookup_uses_row_then_column() -> None:
    depth = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    assert lookup_depth(depth, u=2, v=1) == 7.0


def test_depth_lookup_rejects_negative_index() -> None:
    with pytest.raises(IndexError):
        lookup_depth(np.ones((3, 4), dtype=np.float32), u=-1, v=1)
