'深度モデルに依存しない実験補助処理をまとめます。焦点距離の換算、手指周辺領域の抽出、時系列の連続区間や変化量の集計を担当します。'

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)


# 35mm判換算の焦点距離と画像寸法からピクセル単位の焦点距離を求めます。
def focal_px_from_35mm_equivalent(
    *,
    width: int,
    height: int,
    focal_35mm_mm: float,
) -> float:
    """Approximate pixel focal length from diagonal 35 mm-equivalent FOV."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(focal_35mm_mm) or focal_35mm_mm <= 0:
        raise ValueError("focal_35mm_mm must be finite and positive")
    return focal_35mm_mm * math.hypot(width, height) / _FULL_FRAME_DIAGONAL_MM


# 緑色の矩形領域を画像から検出し、切り出し範囲を返します。
def extract_green_box_roi(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract the central green box without consulting predicted depth."""

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("BGR input must be uint8 with shape (height, width, 3)")
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    candidate = cv2.inRange(hsv, (35, 55, 20), (100, 255, 255))

    # 対象物は意図的に画像の中央付近かつ高さ35%より下に撮影されています。
    # この位置条件で無関係な緑色領域を一定の基準で除外します。
    candidate[: round(height * 0.35), :] = 0
    candidate[:, : round(width * 0.25)] = 0
    candidate[:, round(width * 0.75) :] = 0

    closing_size = max(3, round(min(width, height) * 0.01))
    if closing_size % 2 == 0:
        closing_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_size, closing_size),
    )
    closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, 8)
    if count <= 1:
        raise ValueError("green box segmentation found no component")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[label, cv2.CC_STAT_AREA])
    if area < 100:
        raise ValueError(f"green box component is unexpectedly small: {area} pixels")

    component = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("green box component has no external contour")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    x, y, box_width, box_height = cv2.boundingRect(contour)
    erosion_radius = max(2, round(min(box_width, box_height) * 0.10))
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_radius + 1, 2 * erosion_radius + 1),
    )
    roi = cv2.erode(filled, erosion_kernel)
    if cv2.countNonZero(roi) == 0:
        raise ValueError("green box ROI disappeared after erosion")
    return roi.astype(bool), (x, y, box_width, box_height)


# 関心領域内の深度値を集計し、距離の統計を返します。
def roi_depth_statistics(depth_m: np.ndarray, roi: np.ndarray) -> dict[str, float | int]:
    """Return robust depth statistics within a boolean ROI."""

    if depth_m.shape != roi.shape:
        raise ValueError(f"depth/ROI shapes differ: {depth_m.shape} vs {roi.shape}")
    roi_count = int(np.count_nonzero(roi))
    valid = roi & np.isfinite(depth_m) & (depth_m > 0)
    values = depth_m[valid].astype(np.float64)
    if values.size == 0:
        raise ValueError("green box ROI contains no valid depth")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "roi_pixel_count": roi_count,
        "valid_depth_count": int(values.size),
        "valid_depth_fraction": float(values.size / roi_count),
        "median_m": median,
        "mean_m": float(np.mean(values)),
        "std_m": float(np.std(values)),
        "mad_m": mad,
        "p25_m": float(np.percentile(values, 25)),
        "p75_m": float(np.percentile(values, 75)),
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
    }


# 画像を指定パスへ保存し、書き込み失敗を検出します。
def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


# 指先の深度・座標が有効な時系列行を抽出します。
def _valid_fingertip_rows(records_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            records.append(record)
            fingertips = record.get("fingertips") or []
            if not fingertips:
                continue
            fingertip = fingertips[0]
            depth = fingertip.get("depth_m")
            if fingertip.get("depth_valid") and depth is not None and math.isfinite(depth):
                valid.append(
                    {
                        "frame_index": int(record["frame_index"]),
                        "timestamp_ms": int(record["timestamp_ms"]),
                        "depth_m": float(depth),
                        "u_px": int(fingertip["u_px"]),
                        "v_px": int(fingertip["v_px"]),
                        "width": int(record["width"]),
                        "height": int(record["height"]),
                    }
                )
    return records, valid


# 隣接するフレーム番号を連続区間にまとめます。
def _contiguous_runs(valid: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    for row in valid:
        if not runs or row["frame_index"] != runs[-1][-1]["frame_index"] + 1:
            runs.append([row])
        else:
            runs[-1].append(row)
    return runs


# 欠落フレーム番号を連続区間ごとにまとめます。
def _missing_runs(
    *,
    total_frames: int,
    valid_indices: set[int],
) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    start: int | None = None
    for index in range(total_frames):
        if index not in valid_indices and start is None:
            start = index
        if index in valid_indices and start is not None:
            runs.append({"start_frame": start, "end_frame": index - 1, "length": index - start})
            start = None
    if start is not None:
        runs.append(
            {
                "start_frame": start,
                "end_frame": total_frames - 1,
                "length": total_frames - start,
            }
        )
    return runs


# 隣接フレーム間の移動差分について統計を計算します。
def _delta_statistics(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "max": None,
            "rate_over_0_05_m": None,
            "rate_over_0_10_m": None,
            "rate_over_0_20_m": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "rate_over_0_05_m": float(np.mean(array > 0.05)),
        "rate_over_0_10_m": float(np.mean(array > 0.10)),
        "rate_over_0_20_m": float(np.mean(array > 0.20)),
    }


# 指先の時系列位置から移動量、欠落区間、連続区間を要約します。
def summarize_fingertip_movement(records_path: Path) -> dict[str, Any]:
    """Summarize valid single-pixel fingertip depth without bridging gaps."""

    records, valid = _valid_fingertip_rows(records_path)
    if not records:
        raise ValueError("video records are empty")
    relocalization_threshold_px = 0.10 * math.hypot(
        int(records[0]["width"]),
        int(records[0]["height"]),
    )
    if not valid:
        missing = _missing_runs(total_frames=len(records), valid_indices=set())
        return {
            "processed_frames": len(records),
            "valid_fingertip_depth_frames": 0,
            "valid_fingertip_depth_rate": 0.0,
            "continuous_valid_runs": [],
            "missing_runs": missing,
            "longest_missing_run_frames": max(
                (run["length"] for run in missing),
                default=0,
            ),
            "depth_m": {
                "min": None,
                "p05": None,
                "median": None,
                "p95": None,
                "max": None,
                "p95_minus_p05_movement_range": None,
            },
            "adjacent_valid_frame_absolute_delta_m": _delta_statistics([]),
            "landmark_relocalization_steps": {
                "threshold_px": relocalization_threshold_px,
                "count": 0,
                "steps": [],
            },
            "stable_tracking_adjacent_frame_absolute_delta_m": _delta_statistics([]),
            "absolute_second_difference_m": {"median": None, "p95": None},
            "five_frame_rolling_median_absolute_residual_m": {
                "median": None,
                "p95": None,
            },
            "interpretation_limit": (
                "No fingertip was detected. No movement conclusion can be drawn."
            ),
        }
    runs = _contiguous_runs(valid)
    depths = np.asarray([row["depth_m"] for row in valid], dtype=np.float64)
    deltas: list[float] = []
    stable_tracking_deltas: list[float] = []
    relocalization_steps: list[dict[str, float | int]] = []
    second_differences: list[float] = []
    run_summaries: list[dict[str, Any]] = []
    rolling_residuals: list[float] = []

    for run in runs:
        values = np.asarray([row["depth_m"] for row in run], dtype=np.float64)
        if values.size >= 2:
            for previous, current in pairwise(run):
                depth_delta = abs(current["depth_m"] - previous["depth_m"])
                deltas.append(depth_delta)
                pixel_delta = math.hypot(
                    current["u_px"] - previous["u_px"],
                    current["v_px"] - previous["v_px"],
                )
                if pixel_delta > relocalization_threshold_px:
                    relocalization_steps.append(
                        {
                            "from_frame": previous["frame_index"],
                            "to_frame": current["frame_index"],
                            "pixel_delta": pixel_delta,
                            "depth_delta_m": depth_delta,
                        }
                    )
                else:
                    stable_tracking_deltas.append(depth_delta)
        if values.size >= 3:
            second_differences.extend(np.abs(np.diff(values, n=2)).tolist())
        if values.size >= 5:
            padded = np.pad(values, (2, 2), mode="edge")
            rolling_median = np.asarray(
                [np.median(padded[index : index + 5]) for index in range(values.size)]
            )
            rolling_residuals.extend(np.abs(values - rolling_median).tolist())
        run_summaries.append(
            {
                "start_frame": run[0]["frame_index"],
                "end_frame": run[-1]["frame_index"],
                "start_time_s": run[0]["timestamp_ms"] / 1000.0,
                "end_time_s": run[-1]["timestamp_ms"] / 1000.0,
                "frame_count": len(run),
                "start_depth_m": float(values[0]),
                "end_depth_m": float(values[-1]),
                "min_depth_m": float(np.min(values)),
                "max_depth_m": float(np.max(values)),
                "range_m": float(np.max(values) - np.min(values)),
            }
        )

    second_array = np.asarray(second_differences, dtype=np.float64)
    residual_array = np.asarray(rolling_residuals, dtype=np.float64)
    valid_indices = {row["frame_index"] for row in valid}
    missing = _missing_runs(total_frames=len(records), valid_indices=valid_indices)
    movement_range = float(np.percentile(depths, 95) - np.percentile(depths, 5))
    summary: dict[str, Any] = {
        "processed_frames": len(records),
        "valid_fingertip_depth_frames": len(valid),
        "valid_fingertip_depth_rate": len(valid) / len(records),
        "continuous_valid_runs": run_summaries,
        "missing_runs": missing,
        "longest_missing_run_frames": max((run["length"] for run in missing), default=0),
        "depth_m": {
            "min": float(np.min(depths)),
            "p05": float(np.percentile(depths, 5)),
            "median": float(np.median(depths)),
            "p95": float(np.percentile(depths, 95)),
            "max": float(np.max(depths)),
            "p95_minus_p05_movement_range": movement_range,
        },
        "adjacent_valid_frame_absolute_delta_m": _delta_statistics(deltas),
        "landmark_relocalization_steps": {
            "threshold_px": relocalization_threshold_px,
            "count": len(relocalization_steps),
            "steps": relocalization_steps,
        },
        "stable_tracking_adjacent_frame_absolute_delta_m": _delta_statistics(
            stable_tracking_deltas
        ),
        "absolute_second_difference_m": {
            "median": float(np.median(second_array)) if second_array.size else None,
            "p95": float(np.percentile(second_array, 95)) if second_array.size else None,
        },
        "five_frame_rolling_median_absolute_residual_m": {
            "median": float(np.median(residual_array)) if residual_array.size else None,
            "p95": float(np.percentile(residual_array, 95)) if residual_array.size else None,
        },
        "front_back_direction_validation": (
            "No independent trajectory or direction labels are available; the depth range and "
            "large-scale peaks alone do not prove directional correctness."
        ),
        "interpretation_limit": (
            "The finger is intentionally moving, so deltas/roughness are continuity diagnostics, "
            "not static-scene jitter. No ground-truth trajectory is available."
        ),
    }
    return summary


# 指先深度の時系列をグラフとして保存します。
def write_fingertip_depth_chart(
    *,
    records_path: Path,
    output_path: Path,
) -> None:
    """Render a dependency-free fingertip depth/time plot with OpenCV."""

    records, valid = _valid_fingertip_rows(records_path)
    if not records:
        raise ValueError("cannot plot an empty video record")
    canvas_width, canvas_height = 1600, 900
    if not valid:
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        cv2.putText(
            canvas,
            "No valid fingertip depth",
            (430, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (30, 30, 30),
            3,
            cv2.LINE_AA,
        )
        _write_image(output_path, canvas)
        return
    left, right, top, bottom = 140, 60, 80, 120
    plot_width = canvas_width - left - right
    plot_height = canvas_height - top - bottom
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    max_time = max(int(record["timestamp_ms"]) for record in records) / 1000.0
    time_axis_max = max_time if max_time > 0.0 else 1.0
    depths = np.asarray([row["depth_m"] for row in valid], dtype=np.float64)
    y_min = max(0.0, float(np.min(depths)) - 0.05)
    y_max = float(np.max(depths)) + 0.05
    if y_max <= y_min:
        y_max = y_min + 0.1

    # 時刻と深度をグラフの描画領域内の整数ピクセル座標へ変換します。
    def point(row: dict[str, Any]) -> tuple[int, int]:
        time_s = row["timestamp_ms"] / 1000.0
        x = left + round(time_s / time_axis_max * plot_width)
        y = top + round((y_max - row["depth_m"]) / (y_max - y_min) * plot_height)
        return x, y

    for run in _contiguous_runs(valid):
        points = np.asarray([point(row) for row in run], dtype=np.int32)
        if points.shape[0] >= 2:
            cv2.polylines(canvas, [points], False, (190, 80, 20), 3, cv2.LINE_AA)
        else:
            cv2.circle(canvas, tuple(points[0]), 3, (190, 80, 20), -1, cv2.LINE_AA)

    cv2.rectangle(
        canvas,
        (left, top),
        (left + plot_width, top + plot_height),
        (30, 30, 30),
        2,
    )
    for tick in range(6):
        time_s = time_axis_max * tick / 5
        x = left + round(plot_width * tick / 5)
        cv2.line(canvas, (x, top + plot_height), (x, top + plot_height + 10), (30, 30, 30), 2)
        cv2.putText(
            canvas,
            f"{time_s:.1f}",
            (x - 25, top + plot_height + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        depth = y_max - (y_max - y_min) * tick / 5
        y = top + round(plot_height * tick / 5)
        cv2.line(canvas, (left - 10, y), (left, y), (30, 30, 30), 2)
        cv2.putText(
            canvas,
            f"{depth:.2f}",
            (20, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "Phase 2: INDEX_FINGER_TIP single-pixel depth",
        (left, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "time [s]",
        (left + plot_width // 2 - 50, canvas_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "depth [m]",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    _write_image(output_path, canvas)
