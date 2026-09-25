import importlib.util
import math
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_video_range_report.py"
_SPEC = importlib.util.spec_from_file_location("build_video_range_report_for_test", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load {_SCRIPT_PATH}")
build_video_range_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_video_range_report)


def test_band_metrics_include_boundaries_and_measure_violation() -> None:
    result = build_video_range_report.calculate_band_metrics(
        [0.10, 0.20, 0.25, 0.30, 0.50],
        total_frames=10,
        expected_min_m=0.20,
        expected_max_m=0.30,
    )

    assert result["valid_frames"] == 5
    assert result["valid_rate"] == pytest.approx(0.5)
    assert result["below_range_frames"] == 1
    assert result["below_range_rate"] == pytest.approx(0.2)
    assert result["in_range_frames"] == 3
    assert result["in_range_rate"] == pytest.approx(0.6)
    assert result["above_range_frames"] == 1
    assert result["above_range_rate"] == pytest.approx(0.2)
    violation = result["band_violation_m"]
    assert violation["mean"] == pytest.approx(0.06)
    assert violation["median"] == pytest.approx(0.0)
    assert violation["p95"] == pytest.approx(0.18)
    assert violation["rmse"] == pytest.approx(0.1)
    assert violation["signed_mean"] == pytest.approx(0.02)


def test_band_metrics_signed_mean_preserves_underestimate_direction() -> None:
    result = build_video_range_report.calculate_band_metrics(
        [0.10, 0.15, 0.25],
        total_frames=3,
        expected_min_m=0.20,
        expected_max_m=0.30,
    )

    assert result["band_violation_m"]["signed_mean"] == pytest.approx(-0.05)
    assert result["band_violation_m"]["rmse"] == pytest.approx(
        math.sqrt((0.10**2 + 0.05**2) / 3)
    )


@pytest.mark.parametrize(
    ("depths", "total_frames", "minimum", "maximum"),
    [
        ([], 1, 0.2, 0.3),
        ([0.2], 0, 0.2, 0.3),
        ([0.2], 1, 0.3, 0.2),
        ([float("nan")], 1, 0.2, 0.3),
    ],
)
def test_band_metrics_reject_invalid_inputs(
    depths: list[float], total_frames: int, minimum: float, maximum: float
) -> None:
    with pytest.raises(ValueError):
        build_video_range_report.calculate_band_metrics(
            depths,
            total_frames=total_frames,
            expected_min_m=minimum,
            expected_max_m=maximum,
        )
