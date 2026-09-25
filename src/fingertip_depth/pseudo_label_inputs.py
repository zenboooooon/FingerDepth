"""Prepare immutable RGB and hand-landmark inputs for pseudo-label generation.

This module deliberately stops before teacher inference.  It runs in the root
project environment, where MediaPipe and the canonical OpenCV decoder are
available, and writes a hash-audited interchange dataset that can be consumed
from the isolated Depth Pro environment.
"""

from __future__ import annotations

import hmac
import importlib.metadata
import json
import math
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .artifacts import write_json
from .camera import CameraIntrinsics
from .constants import (
    DEFAULT_FEATURE_LANDMARK_INDICES,
    DEFAULT_TARGET_LANDMARK_INDEX,
    HAND_LANDMARK_NAMES,
)
from .hands import HandLandmarker
from .sample_experiment import focal_px_from_35mm_equivalent
from .video_cache import (
    VideoFrameCache,
    pixel_hash_sequence_sha256,
    pixel_sha256,
    sha256_file,
)

PREPARED_INPUT_FORMAT = "fingertip-depth-pseudo-label-inputs"
PREPARED_INPUT_FORMAT_VERSION = 1
_FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

FrameTransferMode = Literal["copy", "hardlink"]


def _implementation_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parents[1]
    candidates = {
        "pseudo_label_inputs.py": Path(__file__),
        "hands.py": package_dir / "hands.py",
        "constants.py": package_dir / "constants.py",
        "camera.py": package_dir / "camera.py",
        "sample_experiment.py": package_dir / "sample_experiment.py",
        "video_cache.py": package_dir / "video_cache.py",
        "prepare_pseudo_label_inputs.py": project_dir
        / "scripts"
        / "prepare_pseudo_label_inputs.py",
        "pyproject.toml": project_dir / "pyproject.toml",
        "uv.lock": project_dir / "uv.lock",
    }
    return {name: sha256_file(path) for name, path in candidates.items() if path.is_file()}


def _validate_positive_float(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _validate_landmark_indices(indices: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(index) for index in indices)
    if not values:
        raise ValueError("feature_landmark_indices must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("feature_landmark_indices must not contain duplicates")
    invalid = [index for index in values if not 0 <= index < len(HAND_LANDMARK_NAMES)]
    if invalid:
        raise ValueError(f"hand landmark indices are outside 0..20: {invalid}")
    return values


def _validated_sha256(value: str, *, name: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    return digest


def _camera_for_shape(
    *,
    width: int,
    height: int,
    focal_35mm_equivalent_mm: float,
) -> CameraIntrinsics:
    focal_px = focal_px_from_35mm_equivalent(
        width=width,
        height=height,
        focal_35mm_mm=focal_35mm_equivalent_mm,
    )
    return CameraIntrinsics.centered(
        width=width,
        height=height,
        fx_px=focal_px,
    )


def _landmark_dict(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "as_dict"):
        value = observation.as_dict()
    else:
        value = {
            "landmark_index": observation.landmark_index,
            "landmark_name": observation.landmark_name,
            "x_normalized": observation.x_normalized,
            "y_normalized": observation.y_normalized,
            "z_mediapipe_relative": observation.z_mediapipe_relative,
            "u_px": observation.u_px,
            "v_px": observation.v_px,
            "in_frame": observation.in_frame,
        }
    if not isinstance(value, dict):
        raise TypeError("landmark as_dict() must return a dictionary")
    return dict(value)


def _serialize_hand(hand: Any, selected: Sequence[Any]) -> dict[str, Any]:
    return {
        "hand_index": int(hand.hand_index),
        "handedness": hand.handedness,
        "handedness_score": (
            float(hand.handedness_score) if hand.handedness_score is not None else None
        ),
        "landmarks": [_landmark_dict(landmark) for landmark in selected],
    }


def _select_landmarks(
    hand: Any,
    feature_indices: tuple[int, ...],
) -> tuple[list[Any], dict[int, Any]]:
    by_index: dict[int, Any] = {}
    for observation in hand.landmarks:
        index = int(observation.landmark_index)
        if index in by_index:
            raise ValueError(f"duplicate landmark index {index} in MediaPipe result")
        by_index[index] = observation
    return [by_index[index] for index in feature_indices if index in by_index], by_index


def _build_hand_fields(
    hands: Sequence[Any],
    *,
    feature_indices: tuple[int, ...],
    target_index: int,
) -> dict[str, Any]:
    serialized_hands: list[dict[str, Any]] = []
    selected_maps: list[dict[int, Any]] = []
    for hand in hands:
        selected, by_index = _select_landmarks(hand, feature_indices)
        serialized_hands.append(_serialize_hand(hand, selected))
        selected_maps.append(by_index)

    reasons: list[str] = []
    selected_hand_index: int | None = None
    target: dict[str, Any] | None = None
    if not hands:
        reasons.append("no_hand")
    else:
        primary = hands[0]
        selected_hand_index = int(primary.hand_index)
        by_index = selected_maps[0]
        target_observation = by_index.get(target_index)
        if target_observation is None:
            reasons.append("target_landmark_missing")
        else:
            target = _landmark_dict(target_observation)
            if not bool(target_observation.in_frame):
                reasons.append("target_landmark_out_of_frame")
        for index in feature_indices:
            observation = by_index.get(index)
            if observation is None:
                reasons.append(f"feature_landmark_missing:{index}")
            elif not bool(observation.in_frame):
                reasons.append(f"feature_landmark_out_of_frame:{index}")

    return {
        "hand_detected": bool(hands),
        "status": "accepted" if not reasons else "rejected",
        "rejection_reasons": reasons,
        "hands": serialized_hands,
        "selected_hand_index": selected_hand_index,
        "target_landmark": target,
    }


def _write_png(path: Path, bgr: np.ndarray) -> str:
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"failed to write lossless frame: {path}")
    return sha256_file(path)


def _copy_cached_png(
    *,
    source: Path,
    destination: Path,
    mode: FrameTransferMode,
) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:  # pragma: no cover - guarded by public validation
        raise ValueError(f"unsupported frame transfer mode: {mode}")


def _verify_stored_frame(
    *,
    path: Path,
    expected_bgr: np.ndarray,
    expected_pixel_sha256: str,
    expected_png_sha256: str | None = None,
) -> str:
    observed_png_sha256 = sha256_file(path)
    if expected_png_sha256 is not None and not hmac.compare_digest(
        observed_png_sha256,
        expected_png_sha256,
    ):
        raise ValueError(f"stored PNG SHA-256 mismatch: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"stored PNG is not readable: {path}")
    if decoded.shape != expected_bgr.shape:
        raise ValueError(f"stored PNG shape changed: {path}")
    observed_pixel_sha256 = pixel_sha256(decoded)
    if not hmac.compare_digest(observed_pixel_sha256, expected_pixel_sha256):
        raise ValueError(f"stored PNG BGR pixel SHA-256 mismatch: {path}")
    return observed_png_sha256


def _frame_record(
    *,
    frame_index: int,
    timestamp_ms: int,
    relative_path: Path,
    bgr: np.ndarray,
    png_sha256: str,
    bgr_pixel_sha256: str,
    intrinsics: CameraIntrinsics,
    hand_fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "image_path": relative_path.as_posix(),
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "png_sha256": png_sha256,
        "bgr_pixel_sha256": bgr_pixel_sha256,
        "camera_intrinsics": intrinsics.as_dict(),
        **hand_fields,
    }


def _write_record(target: Any, record: dict[str, Any]) -> None:
    target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _video_timestamp_ms(
    *,
    capture: cv2.VideoCapture,
    frame_index: int,
    fps: float,
    previous_timestamp_ms: int | None,
) -> int:
    reported_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    fallback_ms = round(frame_index * 1000.0 / fps)
    if math.isfinite(reported_ms) and reported_ms >= 0 and (frame_index == 0 or reported_ms > 0):
        candidate = round(reported_ms)
    else:
        candidate = fallback_ms
    if previous_timestamp_ms is not None and candidate <= previous_timestamp_ms:
        candidate = previous_timestamp_ms + 1
    return candidate


def _video_decoder_metadata(capture: cv2.VideoCapture) -> dict[str, Any]:
    orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    orientation_metadata_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    orientation_set = False
    orientation_enabled: bool | None = None
    orientation_degrees: float | None = None
    if orientation_property is not None:
        orientation_set = bool(capture.set(orientation_property, 1.0))
        orientation_enabled = capture.get(orientation_property) >= 0.5
    if orientation_metadata_property is not None:
        value = float(capture.get(orientation_metadata_property))
        if math.isfinite(value):
            orientation_degrees = value
    return {
        "opencv_version": cv2.__version__,
        "orientation_auto_set_succeeded": orientation_set,
        "orientation_auto_enabled": orientation_enabled,
        "orientation_metadata_deg": orientation_degrees,
    }


def _load_audited_cache(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[VideoFrameCache, dict[str, Any], Path]:
    expected = _validated_sha256(
        expected_manifest_sha256,
        name="expected_frame_cache_manifest_sha256",
    )
    with manifest_path.open(encoding="utf-8") as source:
        raw_manifest = json.load(source)
    if not isinstance(raw_manifest, dict):
        raise TypeError("frame-cache manifest must be a JSON object")
    source_path = Path(str(raw_manifest.get("source", "")))
    if not source_path.is_file():
        raise FileNotFoundError(f"frame-cache source video was not found: {source_path}")
    baseline_sha256 = str(raw_manifest.get("baseline_records_sha256", ""))
    cache = VideoFrameCache.load(
        manifest_path,
        input_path=source_path,
        baseline_records_sha256=baseline_sha256,
        expected_manifest_sha256=expected,
    )
    cache.verify_all()
    return cache, raw_manifest, source_path


def _cached_png_path(cache: VideoFrameCache, index: int) -> Path:
    entry = cache.manifest["frames"][index]
    path = (cache.manifest_path.parent / entry["relative_path"]).resolve()
    if not path.is_relative_to(cache.manifest_path.parent):
        raise ValueError("cached frame path escapes the frame-cache directory")
    return path


def _ensure_output_directory(output_dir: Path) -> tuple[Path, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"prepared-input output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=False)
    return frames_dir, output_dir / "frames.jsonl"


def prepare_pseudo_label_inputs(
    *,
    output_dir: Path,
    hand_model_path: Path,
    focal_35mm_equivalent_mm: float = 36.0,
    feature_landmark_indices: Sequence[int] = DEFAULT_FEATURE_LANDMARK_INDICES,
    input_video_path: Path | None = None,
    frame_cache_manifest_path: Path | None = None,
    expected_frame_cache_manifest_sha256: str | None = None,
    frame_transfer_mode: FrameTransferMode = "copy",
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Create verified RGB/landmark inputs for the isolated teacher runtime.

    Exactly one of ``input_video_path`` and ``frame_cache_manifest_path`` must
    be supplied.  Every processed frame is retained in ``frames.jsonl``;
    unusable hand detections are explicit rejected records rather than silent
    omissions.
    """

    if (input_video_path is None) == (frame_cache_manifest_path is None):
        raise ValueError("provide exactly one of input_video_path or frame_cache_manifest_path")
    if not hand_model_path.is_file():
        raise FileNotFoundError(f"hand landmarker model was not found: {hand_model_path}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if frame_transfer_mode not in {"copy", "hardlink"}:
        raise ValueError("frame_transfer_mode must be 'copy' or 'hardlink'")
    focal_35mm_equivalent_mm = _validate_positive_float(
        focal_35mm_equivalent_mm,
        name="focal_35mm_equivalent_mm",
    )
    feature_indices = _validate_landmark_indices(feature_landmark_indices)
    target_index = int(DEFAULT_TARGET_LANDMARK_INDEX)
    if not 0 <= target_index < len(HAND_LANDMARK_NAMES):
        raise ValueError("default target landmark index is invalid")

    _frames_dir, records_path = _ensure_output_directory(output_dir)
    records: list[dict[str, Any]] = []
    first_shape: tuple[int, int] | None = None
    first_intrinsics: CameraIntrinsics | None = None
    fps: float
    source_provenance: dict[str, Any]

    def process_frame(
        *,
        detector: HandLandmarker,
        frame_index: int,
        timestamp_ms: int,
        bgr: np.ndarray,
        source_png_path: Path | None = None,
        source_png_sha256: str | None = None,
    ) -> dict[str, Any]:
        nonlocal first_shape, first_intrinsics
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("input frame must be uint8 BGR with shape (H, W, 3)")
        height, width = bgr.shape[:2]
        shape = (height, width)
        if first_shape is None:
            first_shape = shape
            first_intrinsics = _camera_for_shape(
                width=width,
                height=height,
                focal_35mm_equivalent_mm=focal_35mm_equivalent_mm,
            )
        elif shape != first_shape:
            raise ValueError(f"video frame dimensions changed from {first_shape} to {shape}")
        assert first_intrinsics is not None

        relative_path = Path("frames") / f"frame_{frame_index:06d}.png"
        destination = output_dir / relative_path
        expected_pixels = pixel_sha256(bgr)
        if source_png_path is None:
            png_digest = _write_png(destination, bgr)
        else:
            _copy_cached_png(
                source=source_png_path,
                destination=destination,
                mode=frame_transfer_mode,
            )
            png_digest = _verify_stored_frame(
                path=destination,
                expected_bgr=bgr,
                expected_pixel_sha256=expected_pixels,
                expected_png_sha256=source_png_sha256,
            )
        if source_png_path is None:
            _verify_stored_frame(
                path=destination,
                expected_bgr=bgr,
                expected_pixel_sha256=expected_pixels,
                expected_png_sha256=png_digest,
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hands = detector.detect_hands(rgb, timestamp_ms=timestamp_ms)
        hand_fields = _build_hand_fields(
            hands,
            feature_indices=feature_indices,
            target_index=target_index,
        )
        return _frame_record(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            relative_path=relative_path,
            bgr=bgr,
            png_sha256=png_digest,
            bgr_pixel_sha256=expected_pixels,
            intrinsics=first_intrinsics,
            hand_fields=hand_fields,
        )

    with HandLandmarker(model_path=hand_model_path, mode="video", num_hands=1) as detector:
        if input_video_path is not None:
            if not input_video_path.is_file():
                raise FileNotFoundError(f"input video was not found: {input_video_path}")
            capture = cv2.VideoCapture(str(input_video_path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"input is not a readable video: {input_video_path}")
            decoder = _video_decoder_metadata(capture)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                capture.release()
                raise ValueError("video does not report a finite positive FPS")
            previous_timestamp: int | None = None
            try:
                frame_index = 0
                while max_frames is None or frame_index < max_frames:
                    ok, bgr = capture.read()
                    if not ok or bgr is None:
                        break
                    timestamp_ms = _video_timestamp_ms(
                        capture=capture,
                        frame_index=frame_index,
                        fps=fps,
                        previous_timestamp_ms=previous_timestamp,
                    )
                    previous_timestamp = timestamp_ms
                    records.append(
                        process_frame(
                            detector=detector,
                            frame_index=frame_index,
                            timestamp_ms=timestamp_ms,
                            bgr=bgr,
                        )
                    )
                    frame_index += 1
            finally:
                capture.release()
            source_provenance = {
                "mode": "direct_video_decode",
                "video_path": str(input_video_path.resolve()),
                "video_sha256": sha256_file(input_video_path),
                "decoder": decoder,
            }
        else:
            assert frame_cache_manifest_path is not None
            if expected_frame_cache_manifest_sha256 is None:
                raise ValueError("expected_frame_cache_manifest_sha256 is required in cache mode")
            cache, cache_manifest, source_video = _load_audited_cache(
                manifest_path=frame_cache_manifest_path,
                expected_manifest_sha256=expected_frame_cache_manifest_sha256,
            )
            fps = cache.fps
            selected_count = (
                cache.frame_count
                if max_frames is None
                else min(
                    cache.frame_count,
                    max_frames,
                )
            )
            previous_timestamp = None
            for frame_index in range(selected_count):
                entry = cache_manifest["frames"][frame_index]
                if "timestamp_ms" not in entry:
                    raise ValueError(f"frame-cache timestamp is missing at frame {frame_index}")
                timestamp_ms = int(entry["timestamp_ms"])
                if timestamp_ms < 0 or (
                    previous_timestamp is not None and timestamp_ms <= previous_timestamp
                ):
                    raise ValueError("frame-cache timestamps must be strictly increasing")
                previous_timestamp = timestamp_ms
                bgr, observed_pixels = cache.read(frame_index)
                if observed_pixels != entry["bgr_pixel_sha256"]:
                    raise ValueError(f"frame-cache pixel hash mismatch at frame {frame_index}")
                records.append(
                    process_frame(
                        detector=detector,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        bgr=bgr,
                        source_png_path=_cached_png_path(cache, frame_index),
                        source_png_sha256=str(entry["png_sha256"]),
                    )
                )
            source_provenance = {
                "mode": "audited_frame_cache",
                "video_path": str(source_video.resolve()),
                "video_sha256": str(cache_manifest["source_sha256"]),
                "frame_cache_manifest_path": str(frame_cache_manifest_path.resolve()),
                "frame_cache_manifest_sha256": cache.manifest_sha256,
                "expected_frame_cache_manifest_sha256": _validated_sha256(
                    expected_frame_cache_manifest_sha256,
                    name="expected_frame_cache_manifest_sha256",
                ),
                "source_frame_count": cache.frame_count,
                "frame_transfer_mode": frame_transfer_mode,
                "decoder": cache.source_decoder,
            }

    if not records or first_intrinsics is None or first_shape is None:
        raise ValueError("input contained no processable frames")
    with records_path.open("w", encoding="utf-8") as target:
        for record in records:
            _write_record(target, record)

    accepted = sum(record["status"] == "accepted" for record in records)
    no_hand = sum("no_hand" in record["rejection_reasons"] for record in records)
    height, width = first_shape
    manifest: dict[str, Any] = {
        "format": PREPARED_INPUT_FORMAT,
        "format_version": PREPARED_INPUT_FORMAT_VERSION,
        "source": source_provenance,
        "frames_jsonl": {
            "relative_path": "frames.jsonl",
            "sha256": sha256_file(records_path),
        },
        "frames": {
            "relative_directory": "frames",
            "filename_pattern": "frame_%06d.png",
            "pixel_sequence_sha256": pixel_hash_sequence_sha256(
                str(record["bgr_pixel_sha256"]) for record in records
            ),
            "pixel_sequence_encoding": (
                "sha256 of concatenated lowercase hexadecimal BGR pixel SHA-256 "
                "digests encoded as ASCII"
            ),
        },
        "frame_count": len(records),
        "accepted_frame_count": accepted,
        "rejected_frame_count": len(records) - accepted,
        "no_hand_frame_count": no_hand,
        "fps": fps,
        "pixel_format": "uint8 BGR",
        "storage": "lossless PNG",
        "feature_landmarks": {
            "indices": list(feature_indices),
            "names": [HAND_LANDMARK_NAMES[index] for index in feature_indices],
        },
        "target_landmark": {
            "index": target_index,
            "name": HAND_LANDMARK_NAMES[target_index],
            "depth_sampling": "single_pixel",
        },
        "camera": {
            "model": "centered_pinhole_approximation",
            "focal_35mm_equivalent_mm": focal_35mm_equivalent_mm,
            "full_frame_diagonal_mm": _FULL_FRAME_DIAGONAL_MM,
            "formula": "f_px = f_35mm_equivalent_mm * hypot(width_px,height_px) / hypot(36,24)",
            "image_size_px": {"width": width, "height": height},
            "intrinsics": first_intrinsics.as_dict(),
            "constant_across_frames": True,
            "calibrated": False,
        },
        "hand_landmarker": {
            "family": "MediaPipe Hand Landmarker",
            "model_path": str(hand_model_path.resolve()),
            "model_sha256": sha256_file(hand_model_path),
            "mode": "VIDEO",
            "num_hands": 1,
            "min_hand_detection_confidence": 0.5,
            "min_hand_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "processing": {
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "mediapipe_version": importlib.metadata.version("mediapipe"),
            "max_frames": max_frames,
            "teacher_inference_performed": False,
        },
        "provenance": {
            "implementation_sha256": _implementation_hashes(),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def iter_prepared_records(manifest_path: Path) -> Iterable[dict[str, Any]]:
    """Yield records after validating a prepared-input manifest and JSONL hash."""

    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict):
        raise TypeError("prepared-input manifest must be a JSON object")
    if manifest.get("format") != PREPARED_INPUT_FORMAT:
        raise ValueError("unsupported prepared-input format")
    if manifest.get("format_version") != PREPARED_INPUT_FORMAT_VERSION:
        raise ValueError("unsupported prepared-input format version")
    descriptor = manifest.get("frames_jsonl")
    if not isinstance(descriptor, dict):
        raise TypeError("prepared-input manifest has no frames_jsonl descriptor")
    records_path = (manifest_path.parent / str(descriptor.get("relative_path", ""))).resolve()
    if not records_path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("prepared-input records path escapes the dataset directory")
    expected_sha256 = _validated_sha256(str(descriptor.get("sha256", "")), name="records SHA-256")
    if not hmac.compare_digest(sha256_file(records_path), expected_sha256):
        raise ValueError("prepared-input frames.jsonl SHA-256 mismatch")
    with records_path.open(encoding="utf-8") as source:
        for expected_index, line in enumerate(source):
            record = json.loads(line)
            if not isinstance(record, dict) or int(record.get("frame_index", -1)) != expected_index:
                raise ValueError("prepared-input frame indices must be contiguous from zero")
            yield record


__all__ = [
    "PREPARED_INPUT_FORMAT",
    "PREPARED_INPUT_FORMAT_VERSION",
    "FrameTransferMode",
    "iter_prepared_records",
    "prepare_pseudo_label_inputs",
]
