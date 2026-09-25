import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fingertip_depth.sample_experiment import (
    extract_green_box_roi,
    focal_px_from_35mm_equivalent,
    metric3d_scale_audit,
    roi_depth_statistics,
    summarize_fingertip_movement,
    write_fingertip_depth_chart,
)


def test_focal_px_from_diagonal_35mm_equivalent() -> None:
    photo = focal_px_from_35mm_equivalent(width=4284, height=5712, focal_35mm_mm=26)
    video = focal_px_from_35mm_equivalent(width=1440, height=1920, focal_35mm_mm=36)

    assert photo == pytest.approx(4290.606, abs=0.001)
    assert video == pytest.approx(1996.921, abs=0.001)

    audit = metric3d_scale_audit(width=1440, height=1920, focal_px=video)
    assert audit["resize_scale"] == pytest.approx(616 / 1920)
    assert audit["resized_fx_px"] == pytest.approx(640.678727)
    assert audit["canonical_to_metric_factor"] == pytest.approx(0.640678727)
    assert audit["conversion_application_count"] == 1


def test_extract_green_box_roi_joins_faces_and_erodes_boundary() -> None:
    bgr = np.full((600, 400, 3), 180, dtype=np.uint8)
    green = cv2.cvtColor(np.uint8([[[70, 180, 120]]]), cv2.COLOR_HSV2BGR)[0, 0]
    bgr[300:520, 120:280] = green
    bgr[408:411, 120:280] = 255

    roi, bbox = extract_green_box_roi(bgr)

    x, y, width, height = bbox
    assert 115 <= x <= 125
    assert 295 <= y <= 305
    assert width >= 155
    assert height >= 215
    assert roi[350, 200]
    assert not roi[301, 121]


def test_roi_depth_statistics_uses_only_finite_positive_roi() -> None:
    depth = np.asarray([[1.0, 2.0], [np.nan, 0.0]], dtype=np.float32)
    roi = np.ones((2, 2), dtype=bool)

    stats = roi_depth_statistics(depth, roi)

    assert stats["roi_pixel_count"] == 4
    assert stats["valid_depth_count"] == 2
    assert stats["valid_depth_fraction"] == pytest.approx(0.5)
    assert stats["median_m"] == pytest.approx(1.5)


def test_movement_summary_does_not_bridge_detection_gap(tmp_path: Path) -> None:
    records = []
    for index, depth in enumerate((1.0, 1.1, None, 1.4, 1.5)):
        fingertip = (
            []
            if depth is None
            else [
                {
                    "depth_m": depth,
                    "depth_valid": True,
                    "u_px": 10,
                    "v_px": 20,
                }
            ]
        )
        records.append(
            {
                "frame_index": index,
                "timestamp_ms": index * 33,
                "width": 100,
                "height": 100,
                "fingertips": fingertip,
            }
        )
    path = tmp_path / "frames.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_fingertip_movement(path)

    assert summary["valid_fingertip_depth_frames"] == 4
    assert len(summary["continuous_valid_runs"]) == 2
    assert summary["adjacent_valid_frame_absolute_delta_m"]["count"] == 2
    assert summary["longest_missing_run_frames"] == 1


def test_movement_summary_handles_no_detection(tmp_path: Path) -> None:
    records = [
        {
            "frame_index": index,
            "timestamp_ms": index * 33,
            "width": 100,
            "height": 200,
            "fingertips": [],
        }
        for index in range(3)
    ]
    path = tmp_path / "no_detection.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_fingertip_movement(path)
    chart_path = tmp_path / "no_detection.png"
    write_fingertip_depth_chart(records_path=path, output_path=chart_path)

    assert summary["valid_fingertip_depth_frames"] == 0
    assert summary["depth_m"]["median"] is None
    assert summary["adjacent_valid_frame_absolute_delta_m"]["count"] == 0
    assert summary["longest_missing_run_frames"] == 3
    assert chart_path.is_file()


def test_movement_summary_handles_one_valid_depth(tmp_path: Path) -> None:
    records = [
        {
            "frame_index": 0,
            "timestamp_ms": 0,
            "width": 100,
            "height": 200,
            "fingertips": [
                {
                    "depth_m": 0.5,
                    "depth_valid": True,
                    "u_px": 25,
                    "v_px": 50,
                }
            ],
        },
    ]
    path = tmp_path / "one_detection.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_fingertip_movement(path)
    chart_path = tmp_path / "one_detection.png"
    write_fingertip_depth_chart(records_path=path, output_path=chart_path)

    assert summary["valid_fingertip_depth_frames"] == 1
    assert summary["depth_m"]["median"] == pytest.approx(0.5)
    assert summary["adjacent_valid_frame_absolute_delta_m"]["count"] == 0
    assert summary["adjacent_valid_frame_absolute_delta_m"]["max"] is None
    assert chart_path.is_file()
