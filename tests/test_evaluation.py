import numpy as np
import pytest

from fingertip_depth.evaluation import depth_statistics, evaluate_depth


def test_evaluate_depth_ignores_invalid_target_pixels() -> None:
    prediction = np.asarray([[1.0, 3.0], [8.0, 5.0]], dtype=np.float32)
    target = np.asarray([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    metrics = evaluate_depth(prediction, target)
    assert metrics["valid_pixel_count"] == 3
    assert metrics["mae_m"] == pytest.approx(2.0 / 3.0)
    assert metrics["abs_rel"] == pytest.approx(0.25)


def test_depth_statistics_handles_no_valid_depth() -> None:
    stats = depth_statistics(np.asarray([[0.0, np.nan]], dtype=np.float32))
    assert stats["valid_pixel_count"] == 0
    assert stats["median_m"] is None
