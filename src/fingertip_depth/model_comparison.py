"""Fair, camera-condition-controlled comparisons of monocular depth models.

The comparison code deliberately receives an already constructed estimator.  This
keeps model downloads and optional dependency environments outside the experiment
logic, while making the experiment itself straightforward to test without loading
large checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import cv2
import numpy as np

from .artifacts import annotate_fingertip, depth_preview_bgr, write_json
from .camera import CameraIntrinsics
from .coordinates import lookup_depth
from .evaluation import depth_statistics
from .image_io import read_bgr
from .sample_experiment import (
    extract_green_box_roi,
    focal_px_from_35mm_equivalent,
    roi_depth_statistics,
    summarize_fingertip_movement,
    write_fingertip_depth_chart,
)
from .video_cache import VideoFrameCache, pixel_sha256

ConditionId = Literal[
    "unidepth_v2_l__approx_k",
    "unidepth_v2_l__no_camera",
    "depth_pro__approx_focal",
    "depth_pro__estimated_focal",
]
CameraMode = Literal["approx_k", "no_camera", "approx_focal", "estimated_focal"]
PhaseSelection = Literal["all", "phase1", "phase2"]

_CONDITION_DEFINITIONS: dict[str, tuple[str, CameraMode]] = {
    "unidepth_v2_l__approx_k": ("UniDepth V2-L", "approx_k"),
    "unidepth_v2_l__no_camera": ("UniDepth V2-L", "no_camera"),
    "depth_pro__approx_focal": ("Depth Pro", "approx_focal"),
    "depth_pro__estimated_focal": ("Depth Pro", "estimated_focal"),
}


@dataclass(frozen=True, slots=True)
class ComparisonCondition:
    """One of the four pre-registered alternative-model conditions."""

    id: ConditionId
    model_family: str
    camera_mode: CameraMode
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = _CONDITION_DEFINITIONS.get(self.id)
        if expected is None:
            raise ValueError(f"unknown comparison condition: {self.id}")
        if (self.model_family, self.camera_mode) != expected:
            raise ValueError(
                f"condition {self.id} requires model/camera mode {expected}, got "
                f"{(self.model_family, self.camera_mode)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_family": self.model_family,
            "camera_mode": self.camera_mode,
            "metadata": _json_value(dict(self.metadata)),
        }


def comparison_condition(condition_id: ConditionId) -> ComparisonCondition:
    """Construct a validated condition from its stable experiment identifier."""

    model_family, camera_mode = _CONDITION_DEFINITIONS[condition_id]
    return ComparisonCondition(
        id=condition_id,
        model_family=model_family,
        camera_mode=camera_mode,
    )


@runtime_checkable
class PredictionLike(Protocol):
    depth_m: np.ndarray
    inference_ms: float
    device: str


@runtime_checkable
class DepthEstimator(Protocol):
    """Structural interface implemented by the UniDepth and Depth Pro adapters."""

    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: CameraMode,
    ) -> PredictionLike: ...


def sha256_file(path: Path) -> str:
    """Return the lower-case SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # Torch is intentionally not imported by this module.  This handles scalar
    # tensors and small diagnostic arrays returned by optional model adapters.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    return str(value)


def _estimator_metadata(estimator: DepthEstimator) -> dict[str, Any]:
    metadata = getattr(estimator, "metadata", {})
    if callable(metadata):
        metadata = metadata()
    if not isinstance(metadata, Mapping):
        raise TypeError("estimator metadata must be a mapping")
    return _json_value(dict(metadata))


def _prediction_values(
    prediction: PredictionLike,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, float, str, dict[str, Any]]:
    depth_m = np.asarray(prediction.depth_m)
    if depth_m.shape != expected_shape:
        raise ValueError(
            f"model depth shape {depth_m.shape} does not match RGB image {expected_shape}"
        )
    if not np.issubdtype(depth_m.dtype, np.floating):
        raise ValueError(f"model depth must use a floating dtype, got {depth_m.dtype}")
    inference_ms = float(prediction.inference_ms)
    if not math.isfinite(inference_ms) or inference_ms < 0:
        raise ValueError("model inference_ms must be finite and non-negative")
    device = str(prediction.device)
    extras = getattr(prediction, "extras", {})
    if extras is None:
        extras = {}
    if not isinstance(extras, Mapping):
        raise TypeError("prediction extras must be a mapping")
    return (
        np.ascontiguousarray(depth_m.astype(np.float32, copy=False)),
        inference_ms,
        device,
        _json_value(dict(extras)),
    )


def _camera_argument(
    condition: ComparisonCondition,
    approximate_intrinsics: CameraIntrinsics,
) -> CameraIntrinsics | None:
    if condition.camera_mode in {"approx_k", "approx_focal"}:
        return approximate_intrinsics
    return None


def _numeric_leaves(value: Any, *, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaves(child, prefix=child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            leaves.update(_numeric_leaves(child, prefix=child_prefix))
    elif not isinstance(value, bool) and isinstance(value, (int, float, np.number)):
        number = float(value)
        if math.isfinite(number):
            leaves[prefix or "value"] = number
    return leaves


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
            "std": None,
            "coefficient_of_variation": None,
        }
    mean = float(np.mean(array))
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": mean,
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
        "coefficient_of_variation": (
            float(np.std(array) / abs(mean)) if mean != 0.0 else None
        ),
    }


def _summarize_extras(extras: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns: dict[str, list[float]] = {}
    for item in extras:
        for key, value in _numeric_leaves(item).items():
            columns.setdefault(key, []).append(value)
    return {key: _numeric_summary(values) for key, values in sorted(columns.items())}


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
    model_family: str,
) -> np.ndarray:
    annotated = bgr.copy()
    x, y, width, height = bbox
    thickness = max(2, round(min(bgr.shape[:2]) / 700))
    cv2.rectangle(
        annotated,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 255, 255),
        thickness,
    )
    label = f"GT {actual_m:.1f} m | {model_family} ROI median {predicted_m:.3f} m"
    origin = (max(20, x), max(60, y - 20))
    font_scale = max(0.6, min(bgr.shape[:2]) / 2400)
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


def _known_distance_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("known-distance comparison requires at least two samples")
    actual = np.asarray([row["actual_distance_m"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["predicted_depth_m"] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(actual)) or np.any(actual <= 0):
        raise ValueError("all known distances must be finite and positive")
    error = predicted - actual
    slope, intercept = np.polyfit(actual, predicted, 1)
    fitted = slope * actual + intercept
    residual_sum = float(np.sum(np.square(predicted - fitted)))
    total_sum = float(np.sum(np.square(predicted - np.mean(predicted))))
    actual_ranks = np.argsort(np.argsort(actual))
    predicted_ranks = np.argsort(np.argsort(predicted))
    widths = np.asarray([row["bbox_xywh"][2] for row in rows], dtype=np.float64)
    heights = np.asarray([row["bbox_xywh"][3] for row in rows], dtype=np.float64)
    width_products = widths * actual
    height_products = heights * actual
    return {
        "sample_count": len(rows),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "mean_bias_m": float(np.mean(error)),
        "mean_absolute_relative_error": float(np.mean(np.abs(error) / actual)),
        "max_absolute_error_m": float(np.max(np.abs(error))),
        "pearson_r": float(np.corrcoef(actual, predicted)[0, 1]),
        "spearman_r": float(np.corrcoef(actual_ranks, predicted_ranks)[0, 1]),
        "strictly_monotonic_increasing": bool(np.all(np.diff(predicted) > 0)),
        "linear_fit_predicted_from_actual": {
            "slope": float(slope),
            "intercept_m": float(intercept),
            "r_squared": 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0,
        },
        "diagnostic_scale_fit_through_origin": float(
            np.dot(actual, predicted) / np.dot(actual, actual)
        ),
        "scale_fit_applied_to_reported_predictions": False,
        "input_perspective_sanity": {
            "box_width_times_distance_px_m": width_products.tolist(),
            "box_height_times_distance_px_m": height_products.tolist(),
            "box_width_times_distance_cv": float(
                np.std(width_products) / np.mean(width_products)
            ),
            "box_height_times_distance_cv": float(
                np.std(height_products) / np.mean(height_products)
            ),
            "box_width_vs_inverse_distance_pearson_r": float(
                np.corrcoef(widths, 1.0 / actual)[0, 1]
            ),
        },
    }


def evaluate_known_distance_condition(
    *,
    condition: ComparisonCondition,
    estimator: DepthEstimator,
    samples: Sequence[tuple[Path, float]],
    output_dir: Path,
    focal_35mm_mm: float = 26.0,
) -> dict[str, Any]:
    """Evaluate one model/camera condition on the known-distance green box images."""

    if len(samples) < 2:
        raise ValueError("at least two known-distance samples are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    focal_px: float | None = None

    for input_path, actual_value in samples:
        actual_m = float(actual_value)
        if not math.isfinite(actual_m) or actual_m <= 0:
            raise ValueError(f"invalid known distance for {input_path}: {actual_value}")
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
        approximate_intrinsics = CameraIntrinsics.centered(
            width=width,
            height=height,
            fx_px=focal_px,
        )
        prediction = estimator.predict(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            intrinsics=_camera_argument(condition, approximate_intrinsics),
            camera_mode=condition.camera_mode,
        )
        depth_m, inference_ms, device, extras = _prediction_values(
            prediction,
            expected_shape=(height, width),
        )
        roi, bbox = extract_green_box_roi(bgr)
        roi_stats = roi_depth_statistics(depth_m, roi)
        predicted_m = float(roi_stats["median_m"])
        error_m = predicted_m - actual_m
        sample_dir = output_dir / input_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "depth_m.npy", depth_m)
        _write_image(sample_dir / "box_mask.png", roi.astype(np.uint8) * 255)
        _write_image(
            sample_dir / "roi_overlay.jpg",
            _annotate_box(
                bgr,
                bbox=bbox,
                actual_m=actual_m,
                predicted_m=predicted_m,
                model_family=condition.model_family,
            ),
        )
        preview = depth_preview_bgr(depth_m)
        x, y, box_width, box_height = bbox
        cv2.rectangle(
            preview,
            (x, y),
            (x + box_width - 1, y + box_height - 1),
            (255, 255, 255),
            max(2, round(min(bgr.shape[:2]) / 700)),
        )
        _write_image(sample_dir / "depth_preview.png", preview)
        row = {
            "source": str(input_path.resolve()),
            "source_sha256": sha256_file(input_path),
            "input_bgr_pixel_sha256": pixel_sha256(bgr),
            "actual_distance_m": actual_m,
            "predicted_depth_m": predicted_m,
            "signed_error_m": error_m,
            "absolute_error_m": abs(error_m),
            "absolute_relative_error": abs(error_m) / actual_m,
            "bbox_xywh": list(bbox),
            "roi_depth": roi_stats,
            "depth_inference_ms": inference_ms,
            "device": device,
            "prediction_extras": extras,
            "approximate_camera_intrinsics": approximate_intrinsics.as_dict(),
            "camera_information_passed_to_model": condition.camera_mode
            in {"approx_k", "approx_focal"},
        }
        write_json(sample_dir / "result.json", row)
        rows.append(row)

    assert expected_shape is not None and focal_px is not None
    summary: dict[str, Any] = {
        "experiment": "Phase 1 known-distance green-box model comparison",
        "condition": condition.as_dict(),
        "depth_model": _estimator_metadata(estimator),
        "focal_35mm_equivalent_mm": focal_35mm_mm,
        "approximate_focal_px": focal_px,
        "approximate_intrinsics_source": (
            "user-supplied 35mm-equivalent diagonal-FOV approximation; not calibrated K"
        ),
        "camera_information_passed_to_model": condition.camera_mode
        in {"approx_k", "approx_focal"},
        "reported_depth_postprocessing": (
            "native model metric depth; no fitted calibration or scale correction"
        ),
        "representative_depth": "median finite positive model depth inside eroded green-box ROI",
        "roi_method": (
            "fixed HSV H=35..100,S>=55,V>=20; central/lower spatial gate; "
            "1% closing; largest component exterior fill; 10% short-side erosion"
        ),
        "rows": rows,
        "aggregate": _known_distance_aggregate(rows),
        "inference_ms": _numeric_summary([row["depth_inference_ms"] for row in rows]),
        "prediction_extras_numeric_summary": _summarize_extras(
            [row["prediction_extras"] for row in rows]
        ),
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
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


def _validate_expected_sha256(expected_sha256: str) -> str:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected baseline frames SHA-256 must be 64 hexadecimal characters")
    return expected


def _clean_baseline_fingertip(
    fingertip: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    required = {"u_px", "v_px", "landmark_index", "landmark_name"}
    missing = sorted(required - fingertip.keys())
    if missing:
        raise ValueError(f"baseline fingertip is missing fields: {missing}")
    u_px = int(fingertip["u_px"])
    v_px = int(fingertip["v_px"])
    if not 0 <= u_px < width or not 0 <= v_px < height:
        raise ValueError(f"baseline fingertip coordinate {(u_px, v_px)} is out of bounds")
    if int(fingertip["landmark_index"]) != 8:
        raise ValueError("baseline fingertip landmark_index must be 8")
    if str(fingertip["landmark_name"]) != "INDEX_FINGER_TIP":
        raise ValueError("baseline fingertip landmark_name must be INDEX_FINGER_TIP")
    # Depth fields came from the baseline model and must not leak into the new
    # condition.  All coordinate/detection metadata is copied byte-for-value.
    return {
        str(key): _json_value(value)
        for key, value in fingertip.items()
        if key not in {"depth_m", "depth_valid", "depth_error"}
    }


def _load_baseline_records(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    expected = _validate_expected_sha256(expected_sha256)
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            "baseline frames.jsonl SHA-256 mismatch: "
            f"expected {expected}, observed {actual}"
        )
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank line in baseline records at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"baseline line {line_number} is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError("baseline frames.jsonl is empty")
    expected_size: tuple[int, int] | None = None
    previous_timestamp = -1
    for expected_index, record in enumerate(records):
        if int(record.get("frame_index", -1)) != expected_index:
            raise ValueError("baseline frame indices must be contiguous from zero")
        width = int(record.get("width", 0))
        height = int(record.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid baseline dimensions at frame {expected_index}")
        if expected_size is None:
            expected_size = (width, height)
        elif (width, height) != expected_size:
            raise ValueError("baseline frame dimensions changed within the video")
        timestamp = int(record.get("timestamp_ms", -1))
        if timestamp < 0 or timestamp < previous_timestamp:
            raise ValueError("baseline timestamps must be non-negative and monotonic")
        previous_timestamp = timestamp
        hand_detected = record.get("hand_detected")
        if not isinstance(hand_detected, bool):
            raise TypeError(f"baseline hand_detected is not boolean at frame {expected_index}")
        fingertips = record.get("fingertips")
        if not isinstance(fingertips, list) or len(fingertips) > 1:
            raise ValueError("baseline must contain a list with at most one fingertip per frame")
        if hand_detected != bool(fingertips):
            raise ValueError("baseline hand_detected disagrees with fingertip records")
        for fingertip in fingertips:
            if not isinstance(fingertip, Mapping):
                raise TypeError("baseline fingertip record is not an object")
            _clean_baseline_fingertip(fingertip, width=width, height=height)
    return records, actual


def _coordinate_digest(records: Sequence[Mapping[str, Any]]) -> str:
    coordinate_records = []
    for record in records:
        coordinate_records.append(
            {
                "frame_index": record["frame_index"],
                "timestamp_ms": record["timestamp_ms"],
                "width": record["width"],
                "height": record["height"],
                "hand_detected": record["hand_detected"],
                "fingertips": [
                    {
                        str(key): _json_value(value)
                        for key, value in fingertip.items()
                        if key not in {"depth_m", "depth_valid", "depth_error"}
                    }
                    for fingertip in record["fingertips"]
                ],
            }
        )
    encoded = json.dumps(
        coordinate_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_fingertip_depth(
    depth_m: np.ndarray,
    baseline_fingertip: Mapping[str, Any],
) -> tuple[dict[str, Any], float | None]:
    height, width = depth_m.shape
    item = _clean_baseline_fingertip(
        baseline_fingertip,
        width=width,
        height=height,
    )
    try:
        value = lookup_depth(depth_m, int(item["u_px"]), int(item["v_px"]))
    except (IndexError, ValueError) as error:
        item["depth_m"] = None
        item["depth_valid"] = False
        item["depth_error"] = str(error)
        return item, None
    item["depth_m"] = value
    item["depth_valid"] = True
    return item, value


def _relabel_chart(path: Path, condition: ComparisonCondition) -> None:
    chart = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if chart is None:
        raise OSError(f"failed to read generated chart: {path}")
    cv2.rectangle(chart, (125, 5), (1540, 65), (255, 255, 255), cv2.FILLED)
    cv2.putText(
        chart,
        f"Phase 2: INDEX_FINGER_TIP single-pixel {condition.model_family} depth",
        (140, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    _write_image(path, chart)


def evaluate_video_condition(
    *,
    condition: ComparisonCondition,
    estimator: DepthEstimator,
    input_path: Path,
    baseline_records_path: Path,
    baseline_records_sha256: str,
    output_dir: Path,
    frame_cache_manifest_path: Path | None = None,
    frame_cache_manifest_sha256: str | None = None,
    focal_35mm_mm: float = 36.0,
    max_frames: int | None = None,
    write_annotated_video: bool = True,
) -> dict[str, Any]:
    """Evaluate video depth using immutable baseline fingertip coordinates.

    The caller must provide the independently recorded SHA-256 expected for the
    baseline JSONL.  Merely hashing whatever file happens to be present would not
    protect the experiment from silently changing its MediaPipe detections.
    """

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    baseline_records, verified_hash = _load_baseline_records(
        baseline_records_path,
        expected_sha256=baseline_records_sha256,
    )
    selected_records = (
        baseline_records if max_frames is None else baseline_records[:max_frames]
    )
    if not selected_records:
        raise ValueError("no baseline frames selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture: cv2.VideoCapture | None = None
    frame_cache: VideoFrameCache | None = None
    if frame_cache_manifest_path is not None:
        if frame_cache_manifest_sha256 is None:
            raise ValueError("frame_cache_manifest_sha256 is required with a frame cache")
        frame_cache = VideoFrameCache.load(
            frame_cache_manifest_path,
            input_path=input_path,
            baseline_records_sha256=verified_hash,
            expected_manifest_sha256=frame_cache_manifest_sha256,
        )
        if frame_cache.frame_count != len(baseline_records):
            raise ValueError(
                "frame-cache count differs from baseline records: "
                f"{frame_cache.frame_count} vs {len(baseline_records)}"
            )
        fps = frame_cache.fps
        video_decoder: dict[str, Any] = {
            "mode": "lossless_png_frame_cache",
            "manifest_path": str(frame_cache.manifest_path),
            "manifest_sha256": frame_cache.manifest_sha256,
            "source_decoder": frame_cache.source_decoder,
            "consumer_opencv_version": cv2.__version__,
            "png_file_sha256_verified": True,
            "bgr_pixel_sha256_verified": True,
        }
    else:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"input is not a readable video: {input_path}")

        # OpenCV 4 and 5 differ in the default value of ORIENTATION_AUTO.  The
        # iPhone MOV is stored landscape with a 90-degree display transform.
        orientation_auto_requested = False
        orientation_auto_enabled: bool | None = None
        orientation_metadata_deg: float | None = None
        orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        orientation_metadata_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        setter = getattr(capture, "set", None)
        if orientation_property is not None and callable(setter):
            orientation_auto_requested = bool(setter(orientation_property, 1.0))
            orientation_auto_enabled = capture.get(orientation_property) >= 0.5
        if orientation_metadata_property is not None:
            metadata_value = float(capture.get(orientation_metadata_property))
            if math.isfinite(metadata_value):
                orientation_metadata_deg = metadata_value
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError("video does not report a valid FPS")
        video_decoder = {
            "mode": "direct_video_decode",
            "opencv_version": cv2.__version__,
            "orientation_auto_set_succeeded": orientation_auto_requested,
            "orientation_auto_enabled": orientation_auto_enabled,
            "orientation_metadata_deg": orientation_metadata_deg,
        }

    first = selected_records[0]
    width = int(first["width"])
    height = int(first["height"])
    focal_px = focal_px_from_35mm_equivalent(
        width=width,
        height=height,
        focal_35mm_mm=focal_35mm_mm,
    )
    approximate_intrinsics = CameraIntrinsics.centered(
        width=width,
        height=height,
        fx_px=focal_px,
    )
    records_path = output_dir / "frames.jsonl"
    output_video = output_dir / "annotated.mp4"
    writer: cv2.VideoWriter | None = None
    inference_times: list[float] = []
    prediction_extras: list[dict[str, Any]] = []
    devices: set[str] = set()
    valid_fingertip_frames = 0
    decoded_frames = 0

    try:
        with records_path.open("w", encoding="utf-8") as records_file:
            for position, baseline in enumerate(selected_records):
                if frame_cache is not None:
                    bgr, input_pixel_sha256 = frame_cache.read(position)
                else:
                    assert capture is not None
                    success, bgr = capture.read()
                    if not success or bgr is None:
                        raise ValueError(
                            f"video ended after {decoded_frames} frames; baseline requires "
                            f"{len(selected_records)}"
                        )
                    input_pixel_sha256 = None
                decoded_frames += 1
                if bgr.shape[:2] != (int(baseline["height"]), int(baseline["width"])):
                    raise ValueError(
                        f"decoded frame {position} shape {bgr.shape[:2]} disagrees with "
                        f"baseline {(baseline['height'], baseline['width'])}"
                    )
                prediction = estimator.predict(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    intrinsics=_camera_argument(condition, approximate_intrinsics),
                    camera_mode=condition.camera_mode,
                )
                depth_m, inference_ms, device, extras = _prediction_values(
                    prediction,
                    expected_shape=bgr.shape[:2],
                )
                inference_times.append(inference_ms)
                prediction_extras.append(extras)
                devices.add(device)
                record: dict[str, Any] = {
                    "source": str(input_path.resolve()),
                    "frame_index": int(baseline["frame_index"]),
                    "timestamp_ms": int(baseline["timestamp_ms"]),
                    "width": bgr.shape[1],
                    "height": bgr.shape[0],
                    "condition_id": condition.id,
                    "camera_mode": condition.camera_mode,
                    "approximate_camera_intrinsics": approximate_intrinsics.as_dict(),
                    "camera_information_passed_to_model": condition.camera_mode
                    in {"approx_k", "approx_focal"},
                    "depth_inference_ms": inference_ms,
                    "device": device,
                    "prediction_extras": extras,
                    "input_bgr_pixel_sha256": input_pixel_sha256,
                    "depth_statistics": depth_statistics(depth_m),
                    "hand_detected": bool(baseline["hand_detected"]),
                    "fingertips": [],
                }
                if "hand_model" in baseline:
                    record["hand_model"] = _json_value(baseline["hand_model"])
                rendered = bgr
                for baseline_fingertip in baseline["fingertips"]:
                    item, value = _sample_fingertip_depth(depth_m, baseline_fingertip)
                    record["fingertips"].append(item)
                    if value is not None:
                        valid_fingertip_frames += 1
                        rendered = annotate_fingertip(
                            rendered,
                            u_px=int(item["u_px"]),
                            v_px=int(item["v_px"]),
                            depth_m=value,
                            handedness=(
                                str(item["handedness"])
                                if item.get("handedness") is not None
                                else None
                            ),
                        )
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if write_annotated_video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (bgr.shape[1], bgr.shape[0]),
                        )
                        if not writer.isOpened():
                            raise OSError(f"failed to create video writer: {output_video}")
                    writer.write(rendered)

            if max_frames is None and capture is not None:
                success, extra_frame = capture.read()
                if success and extra_frame is not None:
                    raise ValueError(
                        "video contains more decoded frames than the verified baseline records"
                    )
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    movement = summarize_fingertip_movement(records_path)
    chart_path = output_dir / "fingertip_depth_timeseries.png"
    write_fingertip_depth_chart(records_path=records_path, output_path=chart_path)
    _relabel_chart(chart_path, condition)
    hand_frames = sum(bool(record["hand_detected"]) for record in selected_records)
    summary: dict[str, Any] = {
        "experiment": "Phase 2 fingertip front/back model comparison",
        "condition": condition.as_dict(),
        "depth_model": _estimator_metadata(estimator),
        "source": str(input_path.resolve()),
        "source_sha256": sha256_file(input_path),
        "source_fps": fps,
        "video_decoder": video_decoder,
        "processed_frames": decoded_frames,
        "frames_with_hand": hand_frames,
        "hand_detection_rate": hand_frames / decoded_frames,
        "frames_with_valid_fingertip_depth": valid_fingertip_frames,
        "devices": sorted(devices),
        "focal_35mm_equivalent_mm": focal_35mm_mm,
        "approximate_focal_px": focal_px,
        "approximate_camera_intrinsics": approximate_intrinsics.as_dict(),
        "camera_information_passed_to_model": condition.camera_mode
        in {"approx_k", "approx_focal"},
        "approximate_intrinsics_source": (
            "user-supplied 35mm-equivalent diagonal-FOV approximation; not calibrated K; "
            "video stabilization/crop may add scale error"
        ),
        "reported_depth_postprocessing": (
            "native model metric depth; no fitted calibration or scale correction"
        ),
        "baseline_coordinate_source": {
            "path": str(baseline_records_path.resolve()),
            "verified_sha256": verified_hash,
            "expected_sha256": _validate_expected_sha256(baseline_records_sha256),
            "coordinate_detection_sha256": _coordinate_digest(baseline_records),
            "selected_coordinate_detection_sha256": _coordinate_digest(selected_records),
            "full_baseline_frame_count": len(baseline_records),
            "selected_frame_count": len(selected_records),
            "media_pipe_rerun": False,
        },
        "inference_ms": _numeric_summary(inference_times),
        "prediction_extras_numeric_summary": _summarize_extras(prediction_extras),
        "movement_evaluation": movement,
        "artifacts": {
            "frames_jsonl": str(records_path.resolve()),
            "fingertip_depth_timeseries_png": str(chart_path.resolve()),
            "annotated_video": str(output_video.resolve()) if write_annotated_video else None,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def evaluate_condition(
    *,
    condition: ComparisonCondition,
    estimator: DepthEstimator,
    samples: Sequence[tuple[Path, float]],
    video_path: Path,
    baseline_records_path: Path,
    baseline_records_sha256: str | None,
    output_dir: Path,
    frame_cache_manifest_path: Path | None = None,
    frame_cache_manifest_sha256: str | None = None,
    phase: PhaseSelection = "all",
    photo_focal_35mm_mm: float = 26.0,
    video_focal_35mm_mm: float = 36.0,
    max_video_frames: int | None = None,
    write_annotated_video: bool = True,
) -> dict[str, Any]:
    """Run either or both registered experiment phases for one condition."""

    if phase not in {"all", "phase1", "phase2"}:
        raise ValueError("phase must be 'all', 'phase1', or 'phase2'")
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "condition": condition.as_dict(),
        "depth_model": _estimator_metadata(estimator),
        "phase_selection": phase,
    }
    if phase in {"all", "phase1"}:
        result["phase1"] = evaluate_known_distance_condition(
            condition=condition,
            estimator=estimator,
            samples=samples,
            output_dir=output_dir / "phase1_known_distance",
            focal_35mm_mm=photo_focal_35mm_mm,
        )
    if phase in {"all", "phase2"}:
        if baseline_records_sha256 is None:
            raise ValueError("baseline_records_sha256 is required for phase2")
        result["phase2"] = evaluate_video_condition(
            condition=condition,
            estimator=estimator,
            input_path=video_path,
            baseline_records_path=baseline_records_path,
            baseline_records_sha256=baseline_records_sha256,
            output_dir=output_dir / "phase2_finger_movement",
            frame_cache_manifest_path=frame_cache_manifest_path,
            frame_cache_manifest_sha256=frame_cache_manifest_sha256,
            focal_35mm_mm=video_focal_35mm_mm,
            max_frames=max_video_frames,
            write_annotated_video=write_annotated_video,
        )
    # Refresh after lazy loading so checkpoint path and state-dict audit counts
    # are present even in the first condition's root summary.
    result["depth_model"] = _estimator_metadata(estimator)
    write_json(output_dir / "summary.json", result)
    return result


__all__ = [
    "CameraMode",
    "ComparisonCondition",
    "ConditionId",
    "DepthEstimator",
    "PhaseSelection",
    "PredictionLike",
    "comparison_condition",
    "evaluate_condition",
    "evaluate_known_distance_condition",
    "evaluate_video_condition",
    "sha256_file",
]
