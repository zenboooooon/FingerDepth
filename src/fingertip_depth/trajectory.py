'カメラ座標系の指先位置を時系列データとして保持し、軌跡の要約や図表・機械可読ファイルへの出力を支援します。'

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from .geometry import CAMERA_COORDINATE_CONVENTION, CameraPoint3D


# 時刻とカメラ座標系の指先位置からなる軌跡の一点です。
@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One valid fingertip observation in a camera-coordinate trajectory."""

    frame_index: int
    timestamp_ms: int
    u_px: float
    v_px: float
    x_m: float
    y_m: float
    z_m: float

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must not be negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must not be negative")
        if not all(
            math.isfinite(value)
            for value in (self.u_px, self.v_px, self.x_m, self.y_m, self.z_m)
        ):
            raise ValueError("trajectory coordinates must be finite")
        if self.z_m <= 0.0:
            raise ValueError("trajectory z_m must be positive")

    # カメラ座標の三次元点を、フレーム番号・時刻・画素位置付きの軌跡点に変換します。
    @classmethod
    def from_camera_point(
        cls,
        *,
        frame_index: int,
        timestamp_ms: int,
        u_px: float,
        v_px: float,
        point: CameraPoint3D,
    ) -> TrajectoryPoint:
        return cls(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            u_px=float(u_px),
            v_px=float(v_px),
            x_m=point.x_m,
            y_m=point.y_m,
            z_m=point.z_m,
        )

    # 軌跡点のX・Y・Z値からCameraPoint3Dを再構成して返します。
    @property
    def camera_point(self) -> CameraPoint3D:
        return CameraPoint3D(self.x_m, self.y_m, self.z_m)

    # 主要なフィールドを、JSONへ保存できる辞書に変換します。
    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


# 時系列データを系列IDなどの境界ごとの連続区間に分割します。
def contiguous_segments(
    points: Sequence[TrajectoryPoint],
    *,
    max_frame_gap: int = 1,
    max_timestamp_gap_ms: int | None = None,
) -> list[list[TrajectoryPoint]]:
    """Split ordered observations without connecting across missing-frame gaps."""

    if max_frame_gap <= 0:
        raise ValueError("max_frame_gap must be positive")
    if max_timestamp_gap_ms is not None and max_timestamp_gap_ms <= 0:
        raise ValueError("max_timestamp_gap_ms must be positive when supplied")
    if not points:
        return []

    segments: list[list[TrajectoryPoint]] = [[points[0]]]
    for previous, current in pairwise(points):
        frame_gap = current.frame_index - previous.frame_index
        timestamp_gap = current.timestamp_ms - previous.timestamp_ms
        if frame_gap <= 0:
            raise ValueError("trajectory frame indices must be strictly increasing")
        if timestamp_gap <= 0:
            raise ValueError("trajectory timestamps must be strictly increasing")
        contiguous = frame_gap <= max_frame_gap
        if max_timestamp_gap_ms is not None:
            contiguous = contiguous and timestamp_gap <= max_timestamp_gap_ms
        if contiguous:
            segments[-1].append(current)
        else:
            segments.append([current])
    return segments


# 連続区間ごとに軌跡のグループ番号を割り当てます。
def _segment_assignments(
    points: Sequence[TrajectoryPoint],
    *,
    max_frame_gap: int,
    max_timestamp_gap_ms: int | None,
) -> tuple[list[list[TrajectoryPoint]], dict[int, tuple[int, int]]]:
    segments = contiguous_segments(
        points,
        max_frame_gap=max_frame_gap,
        max_timestamp_gap_ms=max_timestamp_gap_ms,
    )
    assignments: dict[int, tuple[int, int]] = {}
    flat_index = 0
    for segment_index, segment in enumerate(segments):
        for point_index, _point in enumerate(segment):
            assignments[flat_index] = (segment_index, point_index)
            flat_index += 1
    return segments, assignments


# 三次元軌跡の各点をCSV形式で保存します。
def write_trajectory_csv(
    path: Path,
    points: Sequence[TrajectoryPoint],
    *,
    max_frame_gap: int = 1,
    max_timestamp_gap_ms: int | None = None,
) -> None:
    """Write an auditable tabular trajectory including segment assignments."""

    _segments, assignments = _segment_assignments(
        points,
        max_frame_gap=max_frame_gap,
        max_timestamp_gap_ms=max_timestamp_gap_ms,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(
            [
                "frame_index",
                "timestamp_ms",
                "u_px",
                "v_px",
                "x_m",
                "y_m",
                "z_m",
                "segment_index",
                "point_index_in_segment",
            ]
        )
        for index, point in enumerate(points):
            segment_index, point_index = assignments[index]
            writer.writerow(
                [
                    point.frame_index,
                    point.timestamp_ms,
                    point.u_px,
                    point.v_px,
                    point.x_m,
                    point.y_m,
                    point.z_m,
                    segment_index,
                    point_index,
                ]
            )


# 三次元軌跡を点群ビューアーで読めるPLY形式で保存します。
def write_trajectory_ply(
    path: Path,
    points: Sequence[TrajectoryPoint],
    *,
    max_frame_gap: int = 1,
    max_timestamp_gap_ms: int | None = None,
) -> None:
    """Write an ASCII PLY whose edges never bridge a missing-frame gap."""

    segments = contiguous_segments(
        points,
        max_frame_gap=max_frame_gap,
        max_timestamp_gap_ms=max_timestamp_gap_ms,
    )
    edges: list[tuple[int, int]] = []
    offset = 0
    for segment in segments:
        edges.extend((offset + index, offset + index + 1) for index in range(len(segment) - 1))
        offset += len(segment)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        target.write("ply\n")
        target.write("format ascii 1.0\n")
        target.write(f"comment camera_coordinates {CAMERA_COORDINATE_CONVENTION}\n")
        target.write(f"element vertex {len(points)}\n")
        target.write("property double x\nproperty double y\nproperty double z\n")
        target.write(f"element edge {len(edges)}\n")
        target.write("property int vertex1\nproperty int vertex2\n")
        target.write("end_header\n")
        for point in points:
            target.write(f"{point.x_m:.17g} {point.y_m:.17g} {point.z_m:.17g}\n")
        for start, end in edges:
            target.write(f"{start} {end}\n")


# 三次元座標を軌跡図のパネル上の画素座標へ変換します。
def _project_to_panel(
    values: np.ndarray,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    margin: int,
) -> np.ndarray:
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    centre = (low + high) / 2.0
    extent = high - low
    extent = np.maximum(extent, np.maximum(np.abs(centre), 1.0) * 1e-6)
    scale = min(
        (width - 2 * margin) / extent[0],
        (height - 2 * margin) / extent[1],
    )
    x = left + width / 2.0 + (values[:, 0] - centre[0]) * scale
    y = top + height / 2.0 + (values[:, 1] - centre[1]) * scale
    return np.rint(np.column_stack((x, y))).astype(np.int32)


# 複数方向から見た軌跡図をPNG画像として保存します。
def write_trajectory_views_png(
    path: Path,
    points: Sequence[TrajectoryPoint],
    *,
    max_frame_gap: int = 1,
    max_timestamp_gap_ms: int | None = None,
    panel_width: int = 600,
    panel_height: int = 600,
) -> None:
    """Render XY, XZ and YZ orthographic views with gaps left disconnected."""

    if panel_width < 240 or panel_height < 240:
        raise ValueError("trajectory view panels must be at least 240x240 pixels")
    segments = contiguous_segments(
        points,
        max_frame_gap=max_frame_gap,
        max_timestamp_gap_ms=max_timestamp_gap_ms,
    )
    canvas = np.full((panel_height, panel_width * 3, 3), 255, dtype=np.uint8)
    panels = (
        ("XY", "X right [m]", "Y down [m]", lambda point: (point.x_m, point.y_m)),
        ("XZ", "X right [m]", "Z forward [m]", lambda point: (point.x_m, point.z_m)),
        ("YZ", "Y down [m]", "Z forward [m]", lambda point: (point.y_m, point.z_m)),
    )
    margin = 70
    if not points:
        cv2.putText(
            canvas,
            "No valid trajectory points",
            (40, panel_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
    else:
        flat_points = list(points)
        for panel_index, (title, x_label, y_label, value_of) in enumerate(panels):
            left = panel_index * panel_width
            cv2.rectangle(
                canvas,
                (left, 0),
                (left + panel_width - 1, panel_height - 1),
                (205, 205, 205),
                1,
            )
            cv2.putText(
                canvas,
                f"{title} trajectory",
                (left + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                x_label,
                (left + 20, panel_height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                y_label,
                (left + 20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            values = np.asarray([value_of(point) for point in flat_points], dtype=np.float64)
            pixels = _project_to_panel(
                values,
                left=left,
                top=0,
                width=panel_width,
                height=panel_height,
                margin=margin,
            )
            offset = 0
            for segment in segments:
                segment_pixels = pixels[offset : offset + len(segment)]
                if len(segment_pixels) >= 2:
                    cv2.polylines(
                        canvas,
                        [segment_pixels.reshape(-1, 1, 2)],
                        False,
                        (210, 105, 30),
                        2,
                        cv2.LINE_AA,
                    )
                for pixel in segment_pixels:
                    cv2.circle(canvas, tuple(pixel), 3, (35, 80, 220), cv2.FILLED, cv2.LINE_AA)
                offset += len(segment)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write trajectory visualization: {path}")


__all__ = [
    "TrajectoryPoint",
    "contiguous_segments",
    "write_trajectory_csv",
    "write_trajectory_ply",
    "write_trajectory_views_png",
]
