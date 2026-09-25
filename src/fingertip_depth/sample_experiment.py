"""Evaluation helpers for the supplied iPhone Phase 1/2 samples."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .artifacts import depth_preview_bgr, write_json
from .camera import CameraIntrinsics
from .constants import (
    METRIC3D_CANONICAL_FOCAL_PX,
    METRIC3D_HUB_MODEL,
    METRIC3D_HUB_REPO,
    METRIC3D_INPUT_HEIGHT,
    METRIC3D_INPUT_WIDTH,
)
from .image_io import read_bgr
from .metric3d import Metric3Dv2, verify_cached_checkpoint
from .pipeline import run_video

_FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def metric3d_scale_audit(
    *,
    width: int,
    height: int,
    focal_px: float,
) -> dict[str, Any]:
    """Describe the single canonical-to-metric conversion used by Metric3D."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(focal_px) or focal_px <= 0:
        raise ValueError("focal_px must be finite and positive")
    resize_scale = min(
        METRIC3D_INPUT_HEIGHT / height,
        METRIC3D_INPUT_WIDTH / width,
    )
    resized_focal_px = focal_px * resize_scale
    return {
        "input_size_px": {"width": width, "height": height},
        "original_fx_px": focal_px,
        "resize_scale": resize_scale,
        "resized_fx_px": resized_focal_px,
        "canonical_focal_px": METRIC3D_CANONICAL_FOCAL_PX,
        "canonical_to_metric_factor": resized_focal_px / METRIC3D_CANONICAL_FOCAL_PX,
        "formula": ("D_metric = D_canonical * (fx_original_px * resize_scale / 1000)"),
        "conversion_implementation": "restore_metric_depth",
        "conversion_application_count": 1,
    }


def extract_green_box_roi(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract the central green box without consulting predicted depth."""

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("BGR input must be uint8 with shape (height, width, 3)")
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    candidate = cv2.inRange(hsv, (35, 55, 20), (100, 255, 255))

    # The target was intentionally photographed near the horizontal centre and
    # below 35% image height. This rejects unrelated green pixels deterministically.
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


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _annotate_box(
    bgr: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    actual_m: float,
    predicted_m: float,
) -> np.ndarray:
    annotated = bgr.copy()
    x, y, width, height = bbox
    thickness = max(3, round(min(bgr.shape[:2]) / 700))
    cv2.rectangle(
        annotated,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 255, 255),
        thickness,
    )
    label = f"GT {actual_m:.1f} m | Metric3D ROI median {predicted_m:.3f} m"
    origin = (max(20, x), max(80, y - 30))
    font_scale = max(0.9, min(bgr.shape[:2]) / 2200)
    cv2.putText(
        annotated,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


def evaluate_known_distance_images(
    *,
    samples: list[tuple[Path, float]],
    output_dir: Path,
    focal_35mm_mm: float = 26.0,
    device: str = "auto",
) -> dict[str, Any]:
    """Run Metric3D once per known-distance image and evaluate the green box."""

    if not samples:
        raise ValueError("at least one known-distance sample is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Metric3Dv2(device=device)
    rows: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    focal_px: float | None = None

    for input_path, actual_m in samples:
        bgr = read_bgr(input_path)
        if bgr is None:
            raise ValueError(f"input is not a readable image: {input_path}")
        height, width = bgr.shape[:2]
        if expected_shape is None:
            expected_shape = (height, width)
            focal_px = focal_px_from_35mm_equivalent(
                width=width,
                height=height,
                focal_35mm_mm=focal_35mm_mm,
            )
        elif (height, width) != expected_shape:
            raise ValueError(
                f"known-distance image dimensions differ: {(height, width)} vs {expected_shape}"
            )
        assert focal_px is not None
        intrinsics = CameraIntrinsics.centered(
            width=width,
            height=height,
            fx_px=focal_px,
        )
        prediction = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), intrinsics)
        roi, bbox = extract_green_box_roi(bgr)
        stats = roi_depth_statistics(prediction.depth_m, roi)
        predicted_m = float(stats["median_m"])
        signed_error_m = predicted_m - actual_m
        sample_dir = output_dir / input_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "depth_m.npy", prediction.depth_m)
        _write_image(sample_dir / "box_mask.png", roi.astype(np.uint8) * 255)
        _write_image(
            sample_dir / "roi_overlay.jpg",
            _annotate_box(
                bgr,
                bbox=bbox,
                actual_m=actual_m,
                predicted_m=predicted_m,
            ),
        )
        preview = depth_preview_bgr(prediction.depth_m)
        x, y, box_width, box_height = bbox
        cv2.rectangle(
            preview,
            (x, y),
            (x + box_width - 1, y + box_height - 1),
            (255, 255, 255),
            max(3, round(min(bgr.shape[:2]) / 700)),
        )
        _write_image(sample_dir / "depth_preview.png", preview)

        row: dict[str, Any] = {
            "source": str(input_path.resolve()),
            "source_sha256": _sha256(input_path),
            "actual_distance_m": actual_m,
            "predicted_depth_m": predicted_m,
            "signed_error_m": signed_error_m,
            "absolute_error_m": abs(signed_error_m),
            "absolute_relative_error": abs(signed_error_m) / actual_m,
            "focal_sensitivity_minus_2_percent_m": predicted_m * 0.98,
            "focal_sensitivity_plus_2_percent_m": predicted_m * 1.02,
            "bbox_xywh": list(bbox),
            "roi_depth": stats,
            "depth_inference_ms": prediction.inference_ms,
            "camera_intrinsics": intrinsics.as_dict(),
            "intrinsics_source": "EXIF 35mm-equivalent diagonal-FOV approximation",
        }
        write_json(sample_dir / "result.json", row)
        rows.append(row)
    assert expected_shape is not None
    assert focal_px is not None

    actual = np.asarray([row["actual_distance_m"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["predicted_depth_m"] for row in rows], dtype=np.float64)
    error = predicted - actual
    slope, intercept = np.polyfit(actual, predicted, 1)
    fitted = slope * actual + intercept
    residual_sum = float(np.sum(np.square(predicted - fitted)))
    total_sum = float(np.sum(np.square(predicted - np.mean(predicted))))
    scale_through_origin = float(np.dot(actual, predicted) / np.dot(actual, actual))
    predicted_ranks = np.argsort(np.argsort(predicted))
    actual_ranks = np.argsort(np.argsort(actual))
    monotonic = bool(np.all(np.diff(predicted) > 0))
    box_widths = np.asarray([row["bbox_xywh"][2] for row in rows], dtype=np.float64)
    box_heights = np.asarray([row["bbox_xywh"][3] for row in rows], dtype=np.float64)
    width_distance_product = box_widths * actual
    height_distance_product = box_heights * actual
    aggregate = {
        "sample_count": len(rows),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "mean_bias_m": float(np.mean(error)),
        "mean_absolute_relative_error": float(np.mean(np.abs(error) / actual)),
        "max_absolute_error_m": float(np.max(np.abs(error))),
        "pearson_r": float(np.corrcoef(actual, predicted)[0, 1]),
        "spearman_r": float(np.corrcoef(actual_ranks, predicted_ranks)[0, 1]),
        "strictly_monotonic_increasing": monotonic,
        "linear_fit_predicted_from_actual": {
            "slope": float(slope),
            "intercept_m": float(intercept),
            "r_squared": 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0,
        },
        "diagnostic_scale_fit_through_origin": scale_through_origin,
        "scale_fit_applied_to_reported_predictions": False,
        "input_perspective_sanity": {
            "box_width_times_distance_px_m": width_distance_product.tolist(),
            "box_height_times_distance_px_m": height_distance_product.tolist(),
            "box_width_times_distance_cv": float(
                np.std(width_distance_product) / np.mean(width_distance_product)
            ),
            "box_height_times_distance_cv": float(
                np.std(height_distance_product) / np.mean(height_distance_product)
            ),
            "box_width_vs_inverse_distance_pearson_r": float(
                np.corrcoef(box_widths, 1.0 / actual)[0, 1]
            ),
        },
    }
    summary: dict[str, Any] = {
        "experiment": "Phase 1 known-distance green-box test",
        "focal_35mm_equivalent_mm": focal_35mm_mm,
        "depth_model": {
            "family": "Metric3D v2",
            "hub_model": METRIC3D_HUB_MODEL,
            "hub_repo": METRIC3D_HUB_REPO,
            "checkpoint_sha256": verify_cached_checkpoint(),
        },
        "device": str(model.device),
        "focal_px": focal_px,
        "metric3d_scale_conversion": metric3d_scale_audit(
            width=expected_shape[1],
            height=expected_shape[0],
            focal_px=focal_px,
        ),
        "intrinsics_source": "EXIF 35mm-equivalent diagonal-FOV approximation; not calibrated K",
        "representative_depth": "median finite positive Metric3D depth inside eroded green-box ROI",
        "roi_method": (
            "fixed HSV H=35..100,S>=55,V>=20; central/lower spatial gate; "
            "1% closing; largest component exterior fill; 10% short-side erosion"
        ),
        "rows": rows,
        "aggregate": aggregate,
    }
    write_json(output_dir / "summary.json", summary)

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image",
                "actual_m",
                "predicted_m",
                "signed_error_m",
                "absolute_error_m",
                "absolute_relative_error",
                "roi_p25_m",
                "roi_p75_m",
                "inference_ms",
            ]
        )
        for row in rows:
            roi_stats = row["roi_depth"]
            writer.writerow(
                [
                    Path(row["source"]).name,
                    row["actual_distance_m"],
                    row["predicted_depth_m"],
                    row["signed_error_m"],
                    row["absolute_error_m"],
                    row["absolute_relative_error"],
                    roi_stats["p25_m"],
                    roi_stats["p75_m"],
                    row["depth_inference_ms"],
                ]
            )
    return summary


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


def _contiguous_runs(valid: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    for row in valid:
        if not runs or row["frame_index"] != runs[-1][-1]["frame_index"] + 1:
            runs.append([row])
        else:
            runs[-1].append(row)
    return runs


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
        "Phase 2: INDEX_FINGER_TIP single-pixel Metric3D depth",
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


def evaluate_finger_movement_video(
    *,
    input_path: Path,
    output_dir: Path,
    hand_model_path: Path,
    focal_35mm_mm: float = 36.0,
    device: str = "auto",
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run Phase 2 and add continuity statistics for the supplied movement video."""

    capture = cv2.VideoCapture(str(input_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"input is not a readable video: {input_path}")
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    focal_px = focal_px_from_35mm_equivalent(
        width=width,
        height=height,
        focal_35mm_mm=focal_35mm_mm,
    )
    phase2_summary = run_video(
        phase=2,
        input_path=input_path,
        output_dir=output_dir,
        fx_px=focal_px,
        fy_px=focal_px,
        device=device,
        frame_step=1,
        max_frames=max_frames,
        save_depth_frames=False,
        hand_model_path=hand_model_path,
    )
    movement = summarize_fingertip_movement(output_dir / "frames.jsonl")
    write_fingertip_depth_chart(
        records_path=output_dir / "frames.jsonl",
        output_path=output_dir / "fingertip_depth_timeseries.png",
    )
    depth_model = dict(phase2_summary["depth_model"])
    depth_model["checkpoint_sha256"] = verify_cached_checkpoint()
    summary = {
        **phase2_summary,
        "depth_model": depth_model,
        "focal_35mm_equivalent_mm": focal_35mm_mm,
        "source_sha256": _sha256(input_path),
        "focal_px": focal_px,
        "metric3d_scale_conversion": metric3d_scale_audit(
            width=width,
            height=height,
            focal_px=focal_px,
        ),
        "intrinsics_source": (
            "user-supplied 35mm-equivalent diagonal-FOV approximation; not calibrated K; "
            "video stabilization/crop may add scale error"
        ),
        "movement_evaluation": movement,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
