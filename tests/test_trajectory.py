import csv
from pathlib import Path

import cv2
import pytest

from fingertip_depth.geometry import CameraPoint3D
from fingertip_depth.trajectory import (
    TrajectoryPoint,
    contiguous_segments,
    write_trajectory_csv,
    write_trajectory_ply,
    write_trajectory_views_png,
)


def _point(frame_index: int, *, timestamp_ms: int | None = None) -> TrajectoryPoint:
    return TrajectoryPoint.from_camera_point(
        frame_index=frame_index,
        timestamp_ms=frame_index * 20 if timestamp_ms is None else timestamp_ms,
        u_px=100.0 + frame_index,
        v_px=200.0 + frame_index,
        point=CameraPoint3D(
            x_m=0.01 * frame_index,
            y_m=-0.02 * frame_index,
            z_m=0.25 + 0.01 * frame_index,
        ),
    )


def test_contiguous_segments_break_at_missing_frames() -> None:
    points = [_point(0), _point(1), _point(3), _point(4)]

    segments = contiguous_segments(points)

    assert [[point.frame_index for point in segment] for segment in segments] == [[0, 1], [3, 4]]


def test_contiguous_segments_can_also_break_on_timestamp_gap() -> None:
    points = [_point(0, timestamp_ms=0), _point(1, timestamp_ms=20), _point(2, timestamp_ms=200)]
    segments = contiguous_segments(points, max_timestamp_gap_ms=50)
    assert [[point.frame_index for point in segment] for segment in segments] == [[0, 1], [2]]


@pytest.mark.parametrize(
    "points",
    [
        [_point(0), _point(0, timestamp_ms=20)],
        [_point(1), _point(0)],
        [_point(0, timestamp_ms=20), _point(1, timestamp_ms=20)],
    ],
)
def test_contiguous_segments_reject_non_increasing_records(
    points: list[TrajectoryPoint],
) -> None:
    with pytest.raises(ValueError):
        contiguous_segments(points)


def test_csv_and_ply_record_segments_without_cross_gap_edges(tmp_path: Path) -> None:
    points = [_point(0), _point(1), _point(3), _point(4)]
    csv_path = tmp_path / "trajectory.csv"
    ply_path = tmp_path / "trajectory.ply"

    write_trajectory_csv(csv_path, points)
    write_trajectory_ply(ply_path, points)

    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert [row["segment_index"] for row in rows] == ["0", "0", "1", "1"]
    assert [row["point_index_in_segment"] for row in rows] == ["0", "1", "0", "1"]

    ply = ply_path.read_text(encoding="utf-8")
    assert "comment camera_coordinates x-right, y-down, z-forward" in ply
    assert "element vertex 4" in ply
    assert "element edge 2" in ply
    assert ply.endswith("0 1\n2 3\n")


def test_projection_png_is_written_for_gapped_trajectory(tmp_path: Path) -> None:
    output = tmp_path / "trajectory_views.png"
    write_trajectory_views_png(output, [_point(0), _point(1), _point(3)])

    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (600, 1800, 3)
    assert image.min() < 255


def test_empty_trajectory_artifacts_are_valid(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    ply_path = tmp_path / "empty.ply"
    png_path = tmp_path / "empty.png"

    write_trajectory_csv(csv_path, [])
    write_trajectory_ply(ply_path, [])
    write_trajectory_views_png(png_path, [])

    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 1
    assert "element vertex 0" in ply_path.read_text(encoding="utf-8")
    assert cv2.imread(str(png_path), cv2.IMREAD_COLOR) is not None


def test_trajectory_point_validates_metadata_and_depth() -> None:
    with pytest.raises(ValueError):
        _point(-1)
    with pytest.raises(ValueError):
        TrajectoryPoint(0, -1, 0.0, 0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        TrajectoryPoint(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
