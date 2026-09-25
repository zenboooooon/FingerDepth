import numpy as np

from fingertip_depth.artifacts import depth_preview_bgr


def test_depth_preview_shape_and_invalid_mask() -> None:
    depth = np.asarray([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    preview = depth_preview_bgr(depth)
    assert preview.shape == (2, 2, 3)
    assert preview.dtype == np.uint8
    assert np.array_equal(preview[0, 0], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(preview[1, 1], np.zeros(3, dtype=np.uint8))
