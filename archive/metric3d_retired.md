# Retired Metric3D code archive

This Markdown archive is intentionally non-importable and non-executable. The Metric3D inference route, its command entry points, its sample evaluators, and their tests were removed from the active project. The original sources are retained below for historical reference only.

The active `constants.py` retains only MediaPipe/shared constants. `experiment_utils.py` contains the model-independent FOV and frame-analysis helpers still used by the Depth Pro/Student workflows.

## `src/fingertip_depth/_metric3d_impl.py`

`````text
"""Pinned Metric3D v2 inference adapter."""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from .camera import CameraIntrinsics
from .constants import (
    METRIC3D_CANONICAL_FOCAL_PX,
    METRIC3D_CHECKPOINT_URL,
    METRIC3D_HUB_MODEL,
    METRIC3D_HUB_REPO,
    METRIC3D_INPUT_HEIGHT,
    METRIC3D_INPUT_WIDTH,
    METRIC3D_MAX_DEPTH_M,
)

_RGB_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_RGB_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
_CHECKPOINT_FILENAME = "metric_depth_vit_small_800k.pth"
_CHECKPOINT_SHA256 = "b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6"


@dataclass(frozen=True, slots=True)
class PreprocessGeometry:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    scale: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


@dataclass(frozen=True, slots=True)
class DepthPrediction:
    depth_m: np.ndarray
    inference_ms: float
    device: str


def prepare_metric3d_input(rgb: np.ndarray) -> tuple[torch.Tensor, PreprocessGeometry]:
    """Apply the exact resize, mean padding and normalization from Metric3D hubconf."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB input must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        raise ValueError("RGB input must use uint8 values")

    original_height, original_width = rgb.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError("RGB input must not be empty")

    scale = min(
        METRIC3D_INPUT_HEIGHT / original_height,
        METRIC3D_INPUT_WIDTH / original_width,
    )
    resized_height = max(1, int(original_height * scale))
    resized_width = max(1, int(original_width * scale))
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_height = METRIC3D_INPUT_HEIGHT - resized_height
    pad_width = METRIC3D_INPUT_WIDTH - resized_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=_RGB_MEAN.tolist(),
    )

    normalized = (padded.astype(np.float32) - _RGB_MEAN) / _RGB_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    geometry = PreprocessGeometry(
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
        scale=scale,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )
    return tensor, geometry


def restore_metric_depth(
    canonical_depth: torch.Tensor,
    *,
    geometry: PreprocessGeometry,
    original_intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Undo padding/resize and de-canonicalize Metric3D depth into metres."""

    depth = canonical_depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Metric3D returned unexpected depth shape {tuple(canonical_depth.shape)}")

    row_end = depth.shape[0] - geometry.pad_bottom if geometry.pad_bottom else depth.shape[0]
    col_end = depth.shape[1] - geometry.pad_right if geometry.pad_right else depth.shape[1]
    depth = depth[
        geometry.pad_top : row_end,
        geometry.pad_left : col_end,
    ]
    if tuple(depth.shape) != (geometry.resized_height, geometry.resized_width):
        raise ValueError(
            "unpadded Metric3D output does not match the preprocessed image: "
            f"got {tuple(depth.shape)}, expected "
            f"{(geometry.resized_height, geometry.resized_width)}"
        )

    depth = torch_f.interpolate(
        depth[None, None],
        size=(geometry.original_height, geometry.original_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Metric3D predicts in a camera with a 1000 px canonical focal length.
    # Resizing the RGB frame also scales its effective focal length.
    scaled_focal_px = original_intrinsics.fx_px * geometry.scale
    depth = depth * (scaled_focal_px / METRIC3D_CANONICAL_FOCAL_PX)
    depth = torch.clamp(depth, min=0.0, max=METRIC3D_MAX_DEPTH_M)
    return depth.detach().to(device="cpu", dtype=torch.float32).numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(path: Path) -> str:
    actual = _sha256(path)
    if actual != _CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Metric3D checkpoint SHA-256 mismatch: expected {_CHECKPOINT_SHA256}, got {actual}"
        )
    return actual


def ensure_cached_checkpoint() -> Path:
    """Download raw bytes, verify them, and only then expose them to torch.load."""

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / _CHECKPOINT_FILENAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        _verify_checkpoint(checkpoint)
        return checkpoint

    with tempfile.NamedTemporaryFile(
        prefix=f"{_CHECKPOINT_FILENAME}.",
        suffix=".download",
        dir=checkpoint.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.hub.download_url_to_file(
            METRIC3D_CHECKPOINT_URL,
            str(temporary_path),
            progress=True,
        )
        _verify_checkpoint(temporary_path)
        temporary_path.replace(checkpoint)
    finally:
        temporary_path.unlink(missing_ok=True)
    return checkpoint


def verify_cached_checkpoint() -> str:
    """Verify the currently cached fixed checkpoint without loading it."""

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / _CHECKPOINT_FILENAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Metric3D checkpoint cache was not created: {checkpoint}")
    return _verify_checkpoint(checkpoint)


class Metric3Dv2:
    """Metric3D v2 ViT-Small fixed at the experiment's phase-1 revision."""

    def __init__(self, *, device: str = "auto") -> None:
        self.device = self._resolve_device(device)
        self._model: Any | None = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError("the pinned Metric3D v2 implementation requires a CUDA GPU")
            return torch.device("cuda")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("the pinned Metric3D v2 implementation only supports CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return resolved

    @property
    def model(self) -> Any:
        if self._model is None:
            checkpoint = ensure_cached_checkpoint()
            model = torch.hub.load(
                METRIC3D_HUB_REPO,
                METRIC3D_HUB_MODEL,
                pretrain=False,
                trust_repo=True,
            )
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(payload["model_state_dict"], strict=False)
            self._model = model.to(self.device).eval()
        return self._model

    def predict(self, rgb: np.ndarray, intrinsics: CameraIntrinsics) -> DepthPrediction:
        tensor, geometry = prepare_metric3d_input(rgb)
        tensor = tensor.to(self.device, non_blocking=True)
        model = self.model

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            canonical_depth, _confidence, _output = model.inference({"input": tensor})
        torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        depth_m = restore_metric_depth(
            canonical_depth,
            geometry=geometry,
            original_intrinsics=intrinsics,
        )
        return DepthPrediction(
            depth_m=depth_m,
            inference_ms=inference_ms,
            device=str(self.device),
        )
`````

## `src/fingertip_depth/metric3d.py`

`````text
"""Public Metric3D v2 adapter with the upstream CUDA-device constraint enforced."""

from __future__ import annotations

import torch

from ._metric3d_impl import (
    DepthPrediction,
    PreprocessGeometry,
    ensure_cached_checkpoint,
    prepare_metric3d_input,
    restore_metric_depth,
    verify_cached_checkpoint,
)
from ._metric3d_impl import (
    Metric3Dv2 as _Metric3Dv2,
)


class Metric3Dv2(_Metric3Dv2):
    """Pinned model restricted to cuda:0 because the upstream decoder hard-codes it."""

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("the pinned Metric3D v2 implementation requires a CUDA GPU")
        resolved = torch.device("cuda:0" if device == "auto" else device)
        if resolved.type != "cuda":
            raise ValueError("the pinned Metric3D v2 implementation only supports CUDA devices")
        if resolved.index not in (None, 0):
            raise ValueError("the pinned Metric3D v2 implementation only supports cuda:0")
        return torch.device("cuda:0")


__all__ = [
    "DepthPrediction",
    "Metric3Dv2",
    "PreprocessGeometry",
    "ensure_cached_checkpoint",
    "prepare_metric3d_input",
    "restore_metric_depth",
    "verify_cached_checkpoint",
]
`````

## `src/fingertip_depth/pipeline.py`

`````text
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
`````

## `src/fingertip_depth/cli.py`

`````text
"""Command-line entry point for phase 1 and phase 2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .pipeline import run_phase1_image, run_phase2_image, run_video

_IMAGE_SUFFIXES = {
    ".bmp",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fx-px",
        type=float,
        required=True,
        help="Calibrated focal length fx in pixels for the original input frame.",
    )
    parser.add_argument("--fy-px", type=float, help="Focal length fy; defaults to fx.")
    parser.add_argument("--cx-px", type=float, help="Principal point cx; defaults to image center.")
    parser.add_argument("--cy-px", type=float, help="Principal point cy; defaults to image center.")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="RGB image or video path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-type",
        choices=("auto", "image", "video"),
        default="auto",
        help="Override image/video detection.",
    )
    parser.add_argument("--device", default="auto", help="auto or cuda:0")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth video frame.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames.")
    parser.add_argument(
        "--save-depth-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save processed video depth maps as .npy. Defaults to on for phase1 and off for phase2."
        ),
    )
    _add_camera_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fingertip-depth",
        description="Metric3D v2 phase-1 depth and phase-2 fingertip-depth PoC.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase1 = subparsers.add_parser("phase1", help="Run Metric3D v2 metric-depth inference.")
    _add_common_arguments(phase1)
    phase1.add_argument("--ground-truth-depth", type=Path, help="Optional GT depth image.")
    phase1.add_argument(
        "--ground-truth-scale",
        type=float,
        default=1.0,
        help="Raw GT units per metre, e.g. 256 for the Metric3D KITTI demo.",
    )

    phase2 = subparsers.add_parser(
        "phase2", help="Run Metric3D v2 plus MediaPipe INDEX_FINGER_TIP lookup."
    )
    _add_common_arguments(phase2)
    phase2.add_argument(
        "--hand-model",
        type=Path,
        default=Path("assets/hand_landmarker.task"),
        help="Path to the fixed MediaPipe Hand Landmarker .task file.",
    )
    return parser


def _input_kind(path: Path, override: str) -> str:
    if override != "auto":
        return override
    return "image" if path.suffix.lower() in _IMAGE_SUFFIXES else "video"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input does not exist or is not a file: {args.input}")
    kind = _input_kind(args.input, args.input_type)
    if kind == "video" and args.command == "phase1" and args.ground_truth_depth is not None:
        parser.error("--ground-truth-depth is only supported for image input")
    camera = {
        "fx_px": args.fx_px,
        "fy_px": args.fy_px,
        "cx_px": args.cx_px,
        "cy_px": args.cy_px,
    }

    try:
        if kind == "video":
            save_depth_frames = args.save_depth_frames
            if save_depth_frames is None:
                save_depth_frames = args.command == "phase1"
            result = run_video(
                phase=1 if args.command == "phase1" else 2,
                input_path=args.input,
                output_dir=args.output_dir,
                device=args.device,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                save_depth_frames=save_depth_frames,
                hand_model_path=args.hand_model if args.command == "phase2" else None,
                **camera,
            )
        elif args.command == "phase1":
            result = run_phase1_image(
                input_path=args.input,
                output_dir=args.output_dir,
                device=args.device,
                ground_truth_depth=args.ground_truth_depth,
                ground_truth_scale=args.ground_truth_scale,
                **camera,
            )
        else:
            result = run_phase2_image(
                input_path=args.input,
                output_dir=args.output_dir,
                hand_model_path=args.hand_model,
                device=args.device,
                **camera,
            )
    except (FileNotFoundError, IndexError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
`````

## `src/fingertip_depth/__main__.py`

`````text
from .cli import main

raise SystemExit(main())
`````

## `src/fingertip_depth/sample_experiment.py`

`````text
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
`````

## `src/fingertip_depth/constants.py`

`````text
"""Fixed experiment and MediaPipe hand-landmark constants."""

FINGERTIP_LANDMARK_INDEX = 8
FINGERTIP_LANDMARK_NAME = "INDEX_FINGER_TIP"

# MediaPipe Hand Landmarker uses this stable 21-landmark topology. Keep the
# tuple index aligned with MediaPipe's landmark index so callers can validate
# serialized feature order without importing MediaPipe itself.
HAND_LANDMARK_NAMES = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

# The initial student-model feature set covers the complete index-finger chain.
# Landmark 8 remains the single-pixel pseudo-label target for backward
# compatibility with phases 1 and 2.
DEFAULT_FEATURE_LANDMARK_INDICES = (5, 6, 7, 8)
DEFAULT_TARGET_LANDMARK_INDEX = FINGERTIP_LANDMARK_INDEX

# Pin both the upstream implementation and the checkpoint selected by hubconf.py.
METRIC3D_HUB_REPO = "YvanYin/Metric3D:eb5b6fac0dc155e4e52f576e304fbf11655ff339"
METRIC3D_HUB_MODEL = "metric3d_vit_small"
METRIC3D_CHECKPOINT_URL = (
    "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth"
)
METRIC3D_INPUT_HEIGHT = 616
METRIC3D_INPUT_WIDTH = 1064
METRIC3D_CANONICAL_FOCAL_PX = 1000.0
METRIC3D_MAX_DEPTH_M = 300.0

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
`````

## `scripts/evaluate_iphone_samples.py`

`````text
"""Run the supplied iPhone known-distance and finger-movement experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fingertip_depth.artifacts import write_json
from fingertip_depth.sample_experiment import (
    evaluate_finger_movement_video,
    evaluate_known_distance_images,
)

_DISTANCE_SAMPLES = (
    ("03image.HEIC", 0.3),
    ("05image.HEIC", 0.5),
    ("07image.HEIC", 0.7),
    ("10image.HEIC", 1.0),
    ("15image.HEIC", 1.5),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("phase1_2_sample"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/iphone_phase1_2"),
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=Path("assets/hand_landmarker.task"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        help=(
            "Phase 2 video path. Defaults to "
            "<input-dir>/finger_movement.MOV when omitted."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("all", "1", "2"), default="all")
    parser.add_argument("--max-video-frames", type=int)
    parser.add_argument(
        "--photo-focal-35mm-mm",
        type=float,
        default=26.0,
        help="Photo 35mm-equivalent focal length from metadata (default: 26).",
    )
    parser.add_argument(
        "--video-focal-35mm-mm",
        type=float,
        default=36.0,
        help="User-supplied video 35mm-equivalent focal length (default: 36).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary: dict[str, object] = {}

    if args.phase in ("all", "1"):
        samples = [(args.input_dir / name, distance) for name, distance in _DISTANCE_SAMPLES]
        missing = [str(path) for path, _distance in samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing known-distance inputs: {missing}")
        print("Phase 1: evaluating five known-distance HEIC images...", flush=True)
        phase1 = evaluate_known_distance_images(
            samples=samples,
            output_dir=args.output_dir / "phase1_known_distance",
            focal_35mm_mm=args.photo_focal_35mm_mm,
            device=args.device,
        )
        summary["phase1_known_distance"] = phase1
        print(json.dumps(phase1["aggregate"], indent=2), flush=True)

    if args.phase in ("all", "2"):
        video = args.video if args.video is not None else args.input_dir / "finger_movement.MOV"
        if not video.is_file():
            raise FileNotFoundError(f"missing movement video: {video}")
        if not args.hand_model.is_file():
            raise FileNotFoundError(f"missing hand model: {args.hand_model}")
        print(f"Phase 2: evaluating {video}...", flush=True)
        phase2 = evaluate_finger_movement_video(
            input_path=video,
            output_dir=args.output_dir / "phase2_finger_movement",
            hand_model_path=args.hand_model,
            focal_35mm_mm=args.video_focal_35mm_mm,
            device=args.device,
            max_frames=args.max_video_frames,
        )
        summary["phase2_finger_movement"] = phase2
        print(json.dumps(phase2["movement_evaluation"], indent=2), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
`````

## `scripts/evaluate_kitti_demo.py`

`````text
"""Evaluate the fixed Metric3D adapter on the official three-image KITTI demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fingertip_depth.artifacts import depth_preview_bgr, write_json
from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.evaluation import evaluate_depth
from fingertip_depth.metric3d import Metric3Dv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Metric3D data/kitti_demo/test_annotations.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    repository_root = args.annotations.parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = Metric3Dv2(device=args.device)
    results: list[dict[str, object]] = []
    all_prediction: list[np.ndarray] = []
    all_target: list[np.ndarray] = []

    for sample in annotation["files"]:
        rgb_path = repository_root / sample["rgb"]
        target_path = repository_root / sample["depth"]
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        target_raw = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or target_raw is None:
            raise ValueError(f"unable to read sample {rgb_path} / {target_path}")
        fx, fy, cx, cy = sample["cam_in"]
        camera = CameraIntrinsics(float(fx), float(fy), float(cx), float(cy))
        prediction = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), camera)
        target_m = target_raw.astype(np.float32) / float(sample["depth_scale"])
        metrics = evaluate_depth(prediction.depth_m, target_m)
        valid = target_m > 0
        all_prediction.append(prediction.depth_m[valid])
        all_target.append(target_m[valid])
        stem = rgb_path.stem
        np.save(args.output_dir / f"{stem}_depth_m.npy", prediction.depth_m)
        cv2.imwrite(
            str(args.output_dir / f"{stem}_depth_preview.png"),
            depth_preview_bgr(prediction.depth_m),
        )
        results.append(
            {
                "image": str(rgb_path.resolve()),
                "inference_ms": prediction.inference_ms,
                "metrics": metrics,
            }
        )

    combined_metrics = evaluate_depth(
        np.concatenate(all_prediction),
        np.concatenate(all_target),
    )
    summary = {
        "sample_count": len(results),
        "samples": results,
        "combined_metrics": combined_metrics,
        "mean_inference_ms": float(np.mean([item["inference_ms"] for item in results])),
        "median_inference_ms": float(np.median([item["inference_ms"] for item in results])),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
`````

## `tests/test_cli.py`

`````text
from pathlib import Path

import pytest

from fingertip_depth import cli


def test_phase1_help_documents_supported_device_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["phase1", "--help"])

    assert exit_info.value.code == 0
    assert "auto or cuda:0" in capsys.readouterr().out


def test_phase1_video_saves_depth_frames_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake video")
    calls: list[dict[str, object]] = []

    def fake_run_video(**kwargs: object) -> dict[str, bool]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "run_video", fake_run_video)

    exit_code = cli.main(
        [
            "phase1",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--fx-px",
            "500",
            "--device",
            "cuda:0",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["phase"] == 1
    assert calls[0]["save_depth_frames"] is True
    assert calls[0]["hand_model_path"] is None


def test_input_kind_auto_detects_heic_image() -> None:
    assert cli._input_kind(Path("frame.HEIC"), "auto") == "image"
`````

## `tests/test_metric3d_device.py`

`````text
import pytest

from fingertip_depth.metric3d import Metric3Dv2


def test_metric3d_restricts_upstream_to_cuda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert str(Metric3Dv2._resolve_device("auto")) == "cuda:0"
    assert str(Metric3Dv2._resolve_device("cuda")) == "cuda:0"
    with pytest.raises(ValueError, match="cuda:0"):
        Metric3Dv2._resolve_device("cuda:1")
    with pytest.raises(ValueError, match="CUDA"):
        Metric3Dv2._resolve_device("cpu")


def test_metric3d_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        Metric3Dv2._resolve_device("auto")
`````

## `tests/test_metric3d_portrait.py`

`````text
import numpy as np
import torch

from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.constants import METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH
from fingertip_depth.metric3d import prepare_metric3d_input, restore_metric_depth


def test_restore_removes_odd_horizontal_padding_from_portrait_input() -> None:
    rgb = np.zeros((960, 641, 3), dtype=np.uint8)
    _tensor, geometry = prepare_metric3d_input(rgb)
    assert geometry.pad_top == geometry.pad_bottom == 0
    assert geometry.pad_right == geometry.pad_left + 1

    canonical = torch.full(
        (1, 1, METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH),
        99.0,
        dtype=torch.float32,
    )
    row_end = METRIC3D_INPUT_HEIGHT - geometry.pad_bottom
    column_end = METRIC3D_INPUT_WIDTH - geometry.pad_right
    canonical[
        :,
        :,
        geometry.pad_top : row_end,
        geometry.pad_left : column_end,
    ] = 2.5
    camera = CameraIntrinsics(
        1000.0 / geometry.scale,
        1000.0 / geometry.scale,
        320.0,
        479.5,
    )

    restored = restore_metric_depth(
        canonical,
        geometry=geometry,
        original_intrinsics=camera,
    )

    assert restored.shape == (960, 641)
    assert np.allclose(restored, 2.5, atol=1e-6)
`````

## `tests/test_metric3d_preprocessing.py`

`````text
import numpy as np
import pytest
import torch

from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.constants import METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH
from fingertip_depth.metric3d import prepare_metric3d_input, restore_metric_depth


def test_preprocess_preserves_ratio_and_nearly_zero_normalizes_padding() -> None:
    rgb = np.full((375, 1242, 3), 64, dtype=np.uint8)
    tensor, geometry = prepare_metric3d_input(rgb)
    assert tuple(tensor.shape) == (1, 3, METRIC3D_INPUT_HEIGHT, METRIC3D_INPUT_WIDTH)
    assert geometry.resized_width == METRIC3D_INPUT_WIDTH
    assert geometry.resized_height == int(375 * (METRIC3D_INPUT_WIDTH / 1242))
    assert geometry.pad_top + geometry.pad_bottom + geometry.resized_height == 616
    # OpenCV rounds the official float padding values when the source is uint8.
    assert torch.all(torch.abs(tensor[0, :, 0, 0]) < 0.01)


def test_restore_unpads_resizes_and_decanonicalizes() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    _tensor, geometry = prepare_metric3d_input(rgb)
    canonical = torch.ones((1, 1, 616, 1064), dtype=torch.float32)
    camera = CameraIntrinsics(800.0, 800.0, 99.5, 49.5)
    restored = restore_metric_depth(
        canonical,
        geometry=geometry,
        original_intrinsics=camera,
    )
    expected = camera.fx_px * geometry.scale / 1000.0
    assert restored.shape == (100, 200)
    assert restored.dtype == np.float32
    assert np.allclose(restored, expected, atol=1e-6)

    doubled_focal = restore_metric_depth(
        canonical,
        geometry=geometry,
        original_intrinsics=CameraIntrinsics(1600.0, 1600.0, 99.5, 49.5),
    )
    assert np.allclose(doubled_focal, restored * 2.0, atol=1e-6)


def test_preprocess_rejects_bgr_like_shape_or_float() -> None:
    with pytest.raises(ValueError):
        prepare_metric3d_input(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        prepare_metric3d_input(np.zeros((10, 10, 3), dtype=np.float32))
`````

## `tests/test_pipeline.py`

`````text
import json
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import pytest

from fingertip_depth import pipeline
from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.hands import FingertipDetection
from fingertip_depth.metric3d import DepthPrediction


def _write_png(path: Path, image: np.ndarray) -> None:
    assert cv2.imwrite(str(path), image)


def _install_fake_metric3d(
    monkeypatch: pytest.MonkeyPatch,
    predictions: list[np.ndarray],
) -> list[tuple[np.ndarray, CameraIntrinsics]]:
    queued = iter(predictions)
    calls: list[tuple[np.ndarray, CameraIntrinsics]] = []

    class FakeMetric3Dv2:
        def __init__(self, *, device: str = "auto") -> None:
            assert device in {"auto", "cpu"}

        def predict(
            self,
            rgb: np.ndarray,
            intrinsics: CameraIntrinsics,
        ) -> DepthPrediction:
            calls.append((rgb.copy(), intrinsics))
            return DepthPrediction(
                depth_m=next(queued).copy(),
                inference_ms=4.25,
                device="cpu",
            )

    monkeypatch.setattr(pipeline, "Metric3Dv2", FakeMetric3Dv2)
    return calls


def _fingertip(*, u_px: int = 3, v_px: int = 1) -> FingertipDetection:
    return FingertipDetection(
        hand_index=0,
        landmark_index=8,
        landmark_name="INDEX_FINGER_TIP",
        x_normalized=0.75,
        y_normalized=0.25,
        u_px=u_px,
        v_px=v_px,
        handedness="Right",
        handedness_score=0.9,
    )


def _install_fake_image_hand_landmarker(
    monkeypatch: pytest.MonkeyPatch,
    detections: list[FingertipDetection],
) -> list[object]:
    instances: list[object] = []

    class FakeHandLandmarker:
        def __init__(self, *, model_path: Path, mode: str, num_hands: int) -> None:
            assert model_path.name == "fake.task"
            assert model_path.is_file()
            assert mode == "image"
            assert num_hands == 1
            self.closed = False
            instances.append(self)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            self.close()

        def close(self) -> None:
            self.closed = True

        def detect(
            self,
            rgb: np.ndarray,
            *,
            timestamp_ms: int | None = None,
        ) -> list[FingertipDetection]:
            assert rgb.dtype == np.uint8
            assert timestamp_ms is None
            return list(detections)

    monkeypatch.setattr(pipeline, "HandLandmarker", FakeHandLandmarker)
    return instances


def test_phase1_image_writes_depth_metadata_and_ground_truth_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bgr = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )
    prediction_m = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_path = tmp_path / "input.png"
    target_path = tmp_path / "target.png"
    output_dir = tmp_path / "phase1"
    _write_png(input_path, bgr)
    _write_png(target_path, (prediction_m * 100).astype(np.uint16))
    calls = _install_fake_metric3d(monkeypatch, [prediction_m])

    record = pipeline.run_phase1_image(
        input_path=input_path,
        output_dir=output_dir,
        fx_px=600.0,
        fy_px=610.0,
        cx_px=1.25,
        cy_px=0.75,
        device="cpu",
        ground_truth_depth=target_path,
        ground_truth_scale=100.0,
    )

    assert len(calls) == 1
    rgb, intrinsics = calls[0]
    assert np.array_equal(rgb, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    assert intrinsics == CameraIntrinsics(600.0, 610.0, 1.25, 0.75)
    assert np.array_equal(np.load(output_dir / "depth_m.npy"), prediction_m)
    assert cv2.imread(str(output_dir / "depth_preview.png")) is not None
    assert record["width"] == 3
    assert record["height"] == 2
    assert record["camera_intrinsics"] == intrinsics.as_dict()
    assert record["ground_truth"]["metrics"] == {
        "valid_pixel_count": 6,
        "mae_m": 0.0,
        "rmse_m": 0.0,
        "abs_rel": 0.0,
    }
    assert json.loads((output_dir / "result.json").read_text(encoding="utf-8")) == record


def test_phase2_image_samples_depth_by_row_then_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bgr = np.full((3, 4, 3), 32, dtype=np.uint8)
    depth_m = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    input_path = tmp_path / "hand.png"
    hand_model_path = tmp_path / "fake.task"
    output_dir = tmp_path / "phase2"
    _write_png(input_path, bgr)
    hand_model_path.write_bytes(b"fake hand landmarker")
    _install_fake_metric3d(monkeypatch, [depth_m])
    instances = _install_fake_image_hand_landmarker(monkeypatch, [_fingertip(u_px=3, v_px=1)])

    record = pipeline.run_phase2_image(
        input_path=input_path,
        output_dir=output_dir,
        hand_model_path=hand_model_path,
        fx_px=500.0,
        device="cpu",
    )

    assert record["hand_detected"] is True
    assert len(record["fingertips"]) == 1
    fingertip = record["fingertips"][0]
    assert fingertip["depth_m"] == 8.0
    assert fingertip["depth_valid"] is True
    assert np.array_equal(np.load(output_dir / "depth_m.npy"), depth_m)
    assert (output_dir / "annotated.png").is_file()
    assert instances and instances[0].closed is True


def test_phase2_image_records_no_hand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bgr = np.full((3, 4, 3), 64, dtype=np.uint8)
    input_path = tmp_path / "no_hand.png"
    hand_model_path = tmp_path / "fake.task"
    output_dir = tmp_path / "phase2_no_hand"
    _write_png(input_path, bgr)
    hand_model_path.write_bytes(b"fake hand landmarker")
    _install_fake_metric3d(monkeypatch, [np.ones((3, 4), dtype=np.float32)])
    _install_fake_image_hand_landmarker(monkeypatch, [])

    record = pipeline.run_phase2_image(
        input_path=input_path,
        output_dir=output_dir,
        hand_model_path=hand_model_path,
        fx_px=500.0,
        device="cpu",
    )

    assert record["hand_detected"] is False
    assert record["fingertips"] == []
    annotated = cv2.imread(str(output_dir / "annotated.png"), cv2.IMREAD_COLOR)
    assert np.array_equal(annotated, bgr)


@pytest.mark.parametrize("invalid_depth", [0.0, np.nan])
def test_phase2_image_records_invalid_fingertip_depth_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_depth: float,
) -> None:
    bgr = np.full((3, 4, 3), 96, dtype=np.uint8)
    depth_m = np.ones((3, 4), dtype=np.float32)
    depth_m[1, 3] = invalid_depth
    input_path = tmp_path / "invalid_depth_hand.png"
    hand_model_path = tmp_path / "fake.task"
    output_dir = tmp_path / "phase2_invalid"
    _write_png(input_path, bgr)
    hand_model_path.write_bytes(b"fake hand landmarker")
    _install_fake_metric3d(monkeypatch, [depth_m])
    _install_fake_image_hand_landmarker(monkeypatch, [_fingertip(u_px=3, v_px=1)])

    record = pipeline.run_phase2_image(
        input_path=input_path,
        output_dir=output_dir,
        hand_model_path=hand_model_path,
        fx_px=500.0,
        device="cpu",
    )

    assert record["hand_detected"] is True
    assert record["fingertips"][0]["depth_m"] is None
    assert record["fingertips"][0]["depth_valid"] is False
    serialized = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert serialized["fingertips"][0]["depth_m"] is None
    assert serialized["fingertips"][0]["depth_valid"] is False
    assert (output_dir / "annotated.png").is_file()


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        timestamps_ms: list[float],
        fps: float,
    ) -> None:
        assert len(frames) == len(timestamps_ms)
        self._frames = frames
        self._timestamps_ms = timestamps_ms
        self._fps = fps
        self._index = -1
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self._index += 1
        if self._index >= len(self._frames):
            return False, None
        return True, self._frames[self._index].copy()

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self._fps
        if property_id == cv2.CAP_PROP_POS_MSEC and 0 <= self._index < len(self._timestamps_ms):
            return self._timestamps_ms[self._index]
        return 0.0

    def release(self) -> None:
        self.released = True


def test_video_frames_prefers_pts_and_honors_frame_step_and_max_frames() -> None:
    frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(5)]
    capture = _FakeCapture(
        frames,
        timestamps_ms=[0.0, 31.2, 74.6, 109.1, 151.8],
        fps=30.0,
    )

    emitted = list(pipeline._video_frames(capture, fps=30.0, frame_step=2, max_frames=2))

    assert [frame.index for frame in emitted] == [0, 2]
    assert [frame.timestamp_ms for frame in emitted] == [0, 75]
    assert [int(frame.bgr[0, 0, 0]) for frame in emitted] == [0, 2]


def test_video_frames_falls_back_to_fps_when_pts_is_stale() -> None:
    frames = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(5)]
    capture = _FakeCapture(
        frames,
        timestamps_ms=[0.0] * len(frames),
        fps=25.0,
    )

    emitted = list(pipeline._video_frames(capture, fps=25.0, frame_step=2, max_frames=None))

    assert [frame.index for frame in emitted] == [0, 2, 4]
    assert [frame.timestamp_ms for frame in emitted] == [0, 80, 160]
`````

## `tests/test_pipeline_video.py`

`````text
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from fingertip_depth import pipeline
from fingertip_depth.camera import CameraIntrinsics
from fingertip_depth.hands import FingertipDetection
from fingertip_depth.metric3d import DepthPrediction


class _FakeCapture:
    def __init__(self, *, fps: float) -> None:
        self.fps = fps
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return self.fps if property_id == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self.released = True


class _FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def _detection() -> FingertipDetection:
    return FingertipDetection(
        hand_index=0,
        landmark_index=8,
        landmark_name="INDEX_FINGER_TIP",
        x_normalized=0.75,
        y_normalized=0.25,
        u_px=3,
        v_px=1,
        handedness="Right",
        handedness_score=0.9,
    )


def test_phase2_video_continues_after_invalid_fingertip_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        pipeline.Frame(
            index=index,
            timestamp_ms=index * 50,
            bgr=np.full((4, 5, 3), 32 + index, dtype=np.uint8),
        )
        for index in range(2)
    ]
    invalid_depth = np.ones((4, 5), dtype=np.float32)
    invalid_depth[1, 3] = 0.0
    valid_depth = np.ones((4, 5), dtype=np.float32)
    valid_depth[1, 3] = 9.0
    predictions = iter([invalid_depth, valid_depth])
    capture = _FakeCapture(fps=20.0)
    writer = _FakeWriter()
    detector_instances: list[object] = []

    class FakeMetric3Dv2:
        def __init__(self, *, device: str) -> None:
            assert device == "cpu"

        def predict(
            self,
            rgb: np.ndarray,
            intrinsics: CameraIntrinsics,
        ) -> DepthPrediction:
            assert rgb.shape == (4, 5, 3)
            assert intrinsics.fx_px == 500.0
            return DepthPrediction(next(predictions).copy(), 2.0, "cpu")

    class FakeHandLandmarker:
        def __init__(self, *, model_path: Path, mode: str, num_hands: int) -> None:
            assert model_path.is_file()
            assert mode == "video"
            assert num_hands == 1
            self.closed = False
            detector_instances.append(self)

        def detect(
            self,
            rgb: np.ndarray,
            *,
            timestamp_ms: int | None = None,
        ) -> list[FingertipDetection]:
            assert rgb.shape == (4, 5, 3)
            assert timestamp_ms in {0, 50}
            return [_detection()]

        def close(self) -> None:
            self.closed = True

    def fake_video_capture(_path: str) -> _FakeCapture:
        return capture

    def fake_video_writer(*_args: object) -> _FakeWriter:
        return writer

    def fake_video_frames(
        _capture: object,
        *,
        fps: float,
        frame_step: int,
        max_frames: int | None,
    ) -> Iterator[pipeline.Frame]:
        assert fps == 20.0
        assert frame_step == 1
        assert max_frames is None
        yield from frames

    monkeypatch.setattr(pipeline, "Metric3Dv2", FakeMetric3Dv2)
    monkeypatch.setattr(pipeline, "HandLandmarker", FakeHandLandmarker)
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", fake_video_capture)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter", fake_video_writer)
    monkeypatch.setattr(pipeline, "_video_frames", fake_video_frames)
    hand_model_path = tmp_path / "fake.task"
    hand_model_path.write_bytes(b"fake hand landmarker")
    output_dir = tmp_path / "video_output"

    summary = pipeline.run_video(
        phase=2,
        input_path=tmp_path / "input.mp4",
        output_dir=output_dir,
        hand_model_path=hand_model_path,
        fx_px=500.0,
        device="cpu",
    )

    records = [
        json.loads(line)
        for line in (output_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert records[0]["input_bgr_pixel_sha256"] == hashlib.sha256(
        np.ascontiguousarray(frames[0].bgr).tobytes()
    ).hexdigest()
    assert records[1]["input_bgr_pixel_sha256"] == hashlib.sha256(
        np.ascontiguousarray(frames[1].bgr).tobytes()
    ).hexdigest()
    assert records[0]["fingertips"][0]["depth_m"] is None
    assert records[0]["fingertips"][0]["depth_valid"] is False
    assert records[1]["fingertips"][0]["depth_m"] == 9.0
    assert records[1]["fingertips"][0]["depth_valid"] is True
    assert summary["processed_frames"] == 2
    assert summary["frames_with_valid_fingertip_depth"] == 1
    assert len(writer.frames) == 2
    assert capture.released is True
    assert writer.released is True
    assert detector_instances and detector_instances[0].closed is True
`````

## `tests/test_sample_experiment.py`

`````text
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
`````

## `tests/test_sample_script.py`

`````text
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_iphone_samples.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_iphone_samples_for_test", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load {_SCRIPT_PATH}")
evaluate_iphone_samples = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_iphone_samples)


def test_phase1_only_does_not_keep_stale_phase2_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for name, _distance in evaluate_iphone_samples._DISTANCE_SAMPLES:
        (input_dir / name).write_bytes(b"fixture")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text(
        json.dumps({"phase2_finger_movement": {"stale": True}}),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {"aggregate": {"ok": True}}

    monkeypatch.setattr(
        evaluate_iphone_samples,
        "evaluate_known_distance_images",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_iphone_samples.py",
            "--phase",
            "1",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--photo-focal-35mm-mm",
            "27.5",
        ],
    )

    assert evaluate_iphone_samples.main() == 0

    combined = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(combined) == {"phase1_known_distance"}
    assert calls[0]["focal_35mm_mm"] == 27.5


def test_phase2_accepts_explicit_video_path(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    custom_video = tmp_path / "finger_movement_2030.MOV"
    custom_video.write_bytes(b"video fixture")
    hand_model = tmp_path / "hand_landmarker.task"
    hand_model.write_bytes(b"model fixture")
    output_dir = tmp_path / "outputs"
    calls: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {"movement_evaluation": {"ok": True}}

    monkeypatch.setattr(
        evaluate_iphone_samples,
        "evaluate_finger_movement_video",
        fake_evaluate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_iphone_samples.py",
            "--phase",
            "2",
            "--input-dir",
            str(input_dir),
            "--video",
            str(custom_video),
            "--hand-model",
            str(hand_model),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert evaluate_iphone_samples.main() == 0
    assert calls[0]["input_path"] == custom_video
`````
