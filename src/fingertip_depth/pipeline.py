"""Phase 1 and phase 2 image/video pipelines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .artifacts import annotate_fingertip, depth_preview_bgr, write_json
from .camera import CameraIntrinsics
from .constants import METRIC3D_HUB_MODEL, METRIC3D_HUB_REPO
from .coordinates import lookup_depth
from .evaluation import depth_statistics, evaluate_depth
from .hands import FingertipDetection, HandLandmarker
from .image_io import read_bgr
from .metric3d import Metric3Dv2

_FIXED_HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    timestamp_ms: int | None
    bgr: np.ndarray


def _read_image(path: Path) -> np.ndarray | None:
    return read_bgr(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hand_model_metadata(path: Path) -> dict[str, object]:
    digest = _sha256(path)
    return {
        "family": "MediaPipe Hand Landmarker",
        "asset_path": str(path.resolve()),
        "sha256": digest,
        "fixed_asset": digest == _FIXED_HAND_MODEL_SHA256,
        "version": "float16/1" if digest == _FIXED_HAND_MODEL_SHA256 else "custom/unknown",
    }


def _camera_for_frame(
    *,
    width: int,
    height: int,
    fx_px: float,
    fy_px: float | None,
    cx_px: float | None,
    cy_px: float | None,
) -> CameraIntrinsics:
    centered = CameraIntrinsics.centered(
        width=width,
        height=height,
        fx_px=fx_px,
        fy_px=fy_px,
    )
    return CameraIntrinsics(
        fx_px=centered.fx_px,
        fy_px=centered.fy_px,
        cx_px=centered.cx_px if cx_px is None else cx_px,
        cy_px=centered.cy_px if cy_px is None else cy_px,
    )


def _base_record(
    *,
    input_path: Path,
    frame: Frame,
    intrinsics: CameraIntrinsics,
    inference_ms: float,
    device: str,
) -> dict[str, object]:
    height, width = frame.bgr.shape[:2]
    return {
        "source": str(input_path.resolve()),
        "frame_index": frame.index,
        "timestamp_ms": frame.timestamp_ms,
        "width": width,
        "height": height,
        "input_bgr_pixel_sha256": hashlib.sha256(
            np.ascontiguousarray(frame.bgr).tobytes()
        ).hexdigest(),
        "camera_intrinsics": intrinsics.as_dict(),
        "depth_model": {
            "family": "Metric3D v2",
            "hub_model": METRIC3D_HUB_MODEL,
            "hub_repo": METRIC3D_HUB_REPO,
        },
        "depth_inference_ms": inference_ms,
        "device": device,
    }


def _fingertip_depth_record(
    depth_m: np.ndarray,
    detection: FingertipDetection,
) -> tuple[dict[str, object], float | None]:
    item = detection.as_dict()
    try:
        value = lookup_depth(depth_m, detection.u_px, detection.v_px)
    except (IndexError, ValueError) as error:
        item["depth_m"] = None
        item["depth_valid"] = False
        item["depth_error"] = str(error)
        return item, None
    item["depth_m"] = value
    item["depth_valid"] = True
    return item, value


def run_phase1_image(
    *,
    input_path: Path,
    output_dir: Path,
    fx_px: float,
    fy_px: float | None = None,
    cx_px: float | None = None,
    cy_px: float | None = None,
    device: str = "auto",
    ground_truth_depth: Path | None = None,
    ground_truth_scale: float = 1.0,
) -> dict[str, object]:
    bgr = _read_image(input_path)
    if bgr is None:
        raise ValueError(f"input is not a readable image: {input_path}")
    height, width = bgr.shape[:2]
    intrinsics = _camera_for_frame(
        width=width,
        height=height,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
    )
    frame = Frame(index=0, timestamp_ms=None, bgr=bgr)
    prediction = Metric3Dv2(device=device).predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), intrinsics)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "depth_m.npy", prediction.depth_m)
    if not cv2.imwrite(
        str(output_dir / "depth_preview.png"), depth_preview_bgr(prediction.depth_m)
    ):
        raise OSError("failed to write depth preview")

    record = _base_record(
        input_path=input_path,
        frame=frame,
        intrinsics=intrinsics,
        inference_ms=prediction.inference_ms,
        device=prediction.device,
    )
    record["depth_statistics"] = depth_statistics(prediction.depth_m)
    if ground_truth_depth is not None:
        if ground_truth_scale <= 0:
            raise ValueError("ground_truth_scale must be positive")
        ground_truth_raw = cv2.imread(str(ground_truth_depth), cv2.IMREAD_UNCHANGED)
        if ground_truth_raw is None:
            raise ValueError(f"unable to read ground-truth depth: {ground_truth_depth}")
        target_m = ground_truth_raw.astype(np.float32) / ground_truth_scale
        record["ground_truth"] = {
            "path": str(ground_truth_depth.resolve()),
            "units_per_metre": ground_truth_scale,
            "metrics": evaluate_depth(prediction.depth_m, target_m),
        }
    write_json(output_dir / "result.json", record)
    return record


def run_phase2_image(
    *,
    input_path: Path,
    output_dir: Path,
    hand_model_path: Path,
    fx_px: float,
    fy_px: float | None = None,
    cx_px: float | None = None,
    cy_px: float | None = None,
    device: str = "auto",
) -> dict[str, object]:
    bgr = _read_image(input_path)
    if bgr is None:
        raise ValueError(f"input is not a readable image: {input_path}")
    height, width = bgr.shape[:2]
    intrinsics = _camera_for_frame(
        width=width,
        height=height,
        fx_px=fx_px,
        fy_px=fy_px,
        cx_px=cx_px,
        cy_px=cy_px,
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prediction = Metric3Dv2(device=device).predict(rgb, intrinsics)
    with HandLandmarker(model_path=hand_model_path, mode="image", num_hands=1) as detector:
        detections = detector.detect(rgb)

    frame = Frame(index=0, timestamp_ms=None, bgr=bgr)
    record = _base_record(
        input_path=input_path,
        frame=frame,
        intrinsics=intrinsics,
        inference_ms=prediction.inference_ms,
        device=prediction.device,
    )
    record["hand_model"] = _hand_model_metadata(hand_model_path)
    record["depth_statistics"] = depth_statistics(prediction.depth_m)
    record["hand_detected"] = bool(detections)
    record["fingertips"] = []
    annotated = bgr
    for detection in detections:
        item, value = _fingertip_depth_record(prediction.depth_m, detection)
        record["fingertips"].append(item)
        if value is not None:
            annotated = annotate_fingertip(
                annotated,
                u_px=detection.u_px,
                v_px=detection.v_px,
                depth_m=value,
                handedness=detection.handedness,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "depth_m.npy", prediction.depth_m)
    if not cv2.imwrite(
        str(output_dir / "depth_preview.png"), depth_preview_bgr(prediction.depth_m)
    ):
        raise OSError("failed to write depth preview")
    if not cv2.imwrite(str(output_dir / "annotated.png"), annotated):
        raise OSError("failed to write annotated image")
    write_json(output_dir / "result.json", record)
    return record


def _video_frames(
    capture: cv2.VideoCapture,
    *,
    fps: float,
    frame_step: int,
    max_frames: int | None,
) -> Iterator[Frame]:
    emitted = 0
    source_index = 0
    previous_timestamp: int | None = None
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        reported_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        fallback_ms = round(source_index * 1000.0 / fps)
        if np.isfinite(reported_ms) and reported_ms >= 0 and (source_index == 0 or reported_ms > 0):
            candidate_ms = round(reported_ms)
        else:
            candidate_ms = fallback_ms
        if previous_timestamp is not None and candidate_ms <= previous_timestamp:
            candidate_ms = previous_timestamp + 1

        if source_index % frame_step == 0:
            previous_timestamp = candidate_ms
            yield Frame(index=source_index, timestamp_ms=candidate_ms, bgr=bgr)
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break
        source_index += 1


def run_video(
    *,
    phase: int,
    input_path: Path,
    output_dir: Path,
    fx_px: float,
    hand_model_path: Path | None = None,
    fy_px: float | None = None,
    cx_px: float | None = None,
    cy_px: float | None = None,
    device: str = "auto",
    frame_step: int = 1,
    max_frames: int | None = None,
    save_depth_frames: bool = False,
) -> dict[str, object]:
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    if phase == 2 and hand_model_path is None:
        raise ValueError("hand_model_path is required for phase 2")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"input is not a readable video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("video does not report a valid FPS")

    writer: cv2.VideoWriter | None = None
    detector: HandLandmarker | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_depth_frames:
            (output_dir / "depth").mkdir(parents=True, exist_ok=True)
        model = Metric3Dv2(device=device)
        hand_metadata: dict[str, object] | None = None
        if phase == 2 and hand_model_path is not None:
            hand_metadata = _hand_model_metadata(hand_model_path)
            detector = HandLandmarker(model_path=hand_model_path, mode="video", num_hands=1)

        records_path = output_dir / "frames.jsonl"
        processed = 0
        detected = 0
        valid_depth_frames = 0
        inference_times: list[float] = []
        with records_path.open("w", encoding="utf-8") as records_file:
            for frame in _video_frames(
                capture,
                fps=fps,
                frame_step=frame_step,
                max_frames=max_frames,
            ):
                height, width = frame.bgr.shape[:2]
                intrinsics = _camera_for_frame(
                    width=width,
                    height=height,
                    fx_px=fx_px,
                    fy_px=fy_px,
                    cx_px=cx_px,
                    cy_px=cy_px,
                )
                rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
                prediction = model.predict(rgb, intrinsics)
                inference_times.append(prediction.inference_ms)
                record = _base_record(
                    input_path=input_path,
                    frame=frame,
                    intrinsics=intrinsics,
                    inference_ms=prediction.inference_ms,
                    device=prediction.device,
                )
                record["depth_statistics"] = depth_statistics(prediction.depth_m)
                rendered = depth_preview_bgr(prediction.depth_m)

                if detector is not None:
                    detections = detector.detect(rgb, timestamp_ms=frame.timestamp_ms)
                    record["hand_model"] = hand_metadata
                    record["hand_detected"] = bool(detections)
                    record["fingertips"] = []
                    if detections:
                        detected += 1
                    rendered = frame.bgr
                    frame_has_valid_depth = False
                    for detection in detections:
                        item, value = _fingertip_depth_record(prediction.depth_m, detection)
                        record["fingertips"].append(item)
                        if value is not None:
                            frame_has_valid_depth = True
                            rendered = annotate_fingertip(
                                rendered,
                                u_px=detection.u_px,
                                v_px=detection.v_px,
                                depth_m=value,
                                handedness=detection.handedness,
                            )
                    if frame_has_valid_depth:
                        valid_depth_frames += 1

                if save_depth_frames:
                    np.save(
                        output_dir / "depth" / f"frame_{frame.index:06d}_m.npy",
                        prediction.depth_m,
                    )
                if writer is None:
                    output_video = output_dir / (
                        "annotated.mp4" if phase == 2 else "depth_preview.mp4"
                    )
                    writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps / frame_step,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise OSError(f"failed to create video writer: {output_video}")
                writer.write(rendered)
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

        if processed == 0:
            raise ValueError("video contained no processable frames")
        summary: dict[str, object] = {
            "phase": phase,
            "source": str(input_path.resolve()),
            "processed_frames": processed,
            "source_fps": fps,
            "frame_step": frame_step,
            "mean_depth_inference_ms": float(np.mean(inference_times)),
            "median_depth_inference_ms": float(np.median(inference_times)),
            "device": prediction.device,
            "depth_model": {
                "family": "Metric3D v2",
                "hub_model": METRIC3D_HUB_MODEL,
                "hub_repo": METRIC3D_HUB_REPO,
            },
        }
        if phase == 2:
            summary["hand_model"] = hand_metadata
            summary["frames_with_hand"] = detected
            summary["hand_detection_rate"] = detected / processed
            summary["frames_with_valid_fingertip_depth"] = valid_depth_frames
        write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if detector is not None:
            detector.close()
