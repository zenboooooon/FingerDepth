"""MediaPipe hand landmarks with a backward-compatible landmark-8 view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

from .constants import (
    FINGERTIP_LANDMARK_INDEX,
    FINGERTIP_LANDMARK_NAME,
    HAND_LANDMARK_NAMES,
)
from .coordinates import normalized_to_pixel

_HAND_LANDMARK_INDEX_BY_NAME = {
    name: index for index, name in enumerate(HAND_LANDMARK_NAMES)
}


def parse_hand_landmark_selection(value: str) -> tuple[int, ...]:
    """Parse ordered comma-separated MediaPipe landmark names or indices.

    Names are matched case-insensitively. ``all`` is accepted only by itself
    and expands to the official 21-landmark order. Duplicate indices are
    rejected rather than silently changing a model feature vector.
    """

    if not isinstance(value, str):
        raise TypeError("hand landmark selection must be a string")
    if not value.strip():
        raise ValueError("hand landmark selection must not be empty")

    tokens = [token.strip() for token in value.split(",")]
    if any(not token for token in tokens):
        raise ValueError("hand landmark selection contains an empty item")
    if any(token.casefold() == "all" for token in tokens):
        if len(tokens) != 1:
            raise ValueError("'all' must be used by itself")
        return tuple(range(len(HAND_LANDMARK_NAMES)))

    indices: list[int] = []
    for token in tokens:
        try:
            index = int(token, 10)
        except ValueError:
            normalized_name = token.upper()
            try:
                index = _HAND_LANDMARK_INDEX_BY_NAME[normalized_name]
            except KeyError as error:
                raise ValueError(f"unknown hand landmark: {token!r}") from error
        if not 0 <= index < len(HAND_LANDMARK_NAMES):
            raise ValueError(
                f"hand landmark index must be in [0, {len(HAND_LANDMARK_NAMES) - 1}]: "
                f"{index}"
            )
        if index in indices:
            raise ValueError(f"duplicate hand landmark index: {index}")
        indices.append(index)
    return tuple(indices)


@dataclass(frozen=True, slots=True)
class FingertipDetection:
    hand_index: int
    landmark_index: int
    landmark_name: str
    x_normalized: float
    y_normalized: float
    u_px: int
    v_px: int
    handedness: str | None
    handedness_score: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LandmarkObservation:
    """One MediaPipe landmark in normalized and original-image coordinates.

    ``z_mediapipe_relative`` is MediaPipe's relative hand-landmark value. It
    is deliberately not named ``depth`` because it is not metric camera depth.
    Pixel coordinates are absent when the normalized point lies outside the
    image; the normalized observation is retained for audit and filtering.
    """

    landmark_index: int
    landmark_name: str
    x_normalized: float
    y_normalized: float
    z_mediapipe_relative: float
    u_px: int | None
    v_px: int | None
    in_frame: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HandDetection:
    """All available MediaPipe landmarks for one detected hand."""

    hand_index: int
    handedness: str | None
    handedness_score: float | None
    landmarks: tuple[LandmarkObservation, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "hand_index": self.hand_index,
            "handedness": self.handedness,
            "handedness_score": self.handedness_score,
            "landmarks": [landmark.as_dict() for landmark in self.landmarks],
        }

    def landmark(self, index: int) -> LandmarkObservation | None:
        """Return one indexed observation, or ``None`` when it was unavailable."""

        return next(
            (landmark for landmark in self.landmarks if landmark.landmark_index == index),
            None,
        )


class HandLandmarker:
    """Synchronous IMAGE/VIDEO MediaPipe Tasks wrapper."""

    def __init__(
        self,
        *,
        model_path: Path,
        mode: Literal["image", "video"],
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe Hand Landmarker model not found: {model_path}. "
                "Run scripts/download_hand_landmarker.py first."
            )
        if mode not in ("image", "video"):
            raise ValueError("mode must be 'image' or 'video'")
        if num_hands <= 0:
            raise ValueError("num_hands must be positive")

        import mediapipe as mp

        self._mp = mp
        self._mode = mode
        self._last_timestamp_ms: int | None = None
        running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if mode == "image"
            else mp.tasks.vision.RunningMode.VIDEO
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def detect(
        self,
        rgb: np.ndarray,
        *,
        timestamp_ms: int | None = None,
    ) -> list[FingertipDetection]:
        """Return the legacy landmark-8-only representation."""

        result, width, height = self._run_detection(rgb, timestamp_ms=timestamp_ms)
        return self._extract(result, width=width, height=height)

    def detect_hands(
        self,
        rgb: np.ndarray,
        *,
        timestamp_ms: int | None = None,
    ) -> list[HandDetection]:
        """Return all available official landmarks for every detected hand."""

        result, width, height = self._run_detection(rgb, timestamp_ms=timestamp_ms)
        return self._extract_hands(result, width=width, height=height)

    def _run_detection(
        self,
        rgb: np.ndarray,
        *,
        timestamp_ms: int | None,
    ) -> tuple[Any, int, int]:
        """Run MediaPipe exactly once for one public detection call."""

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB input must be uint8 with shape (height, width, 3)")
        height, width = rgb.shape[:2]
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        if self._mode == "image":
            if timestamp_ms is not None:
                raise ValueError("timestamp_ms is only valid in video mode")
            result = self._landmarker.detect(mp_image)
        else:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in video mode")
            if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
                raise ValueError("video timestamps must be strictly increasing")
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            self._last_timestamp_ms = timestamp_ms

        return result, width, height

    @staticmethod
    def next_video_timestamp_ms(
        frame_index: int,
        fps: float,
        previous_timestamp_ms: int | None,
    ) -> int:
        if frame_index < 0:
            raise ValueError("frame_index must not be negative")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        candidate = round(frame_index * 1000.0 / fps)
        if previous_timestamp_ms is not None and candidate <= previous_timestamp_ms:
            return previous_timestamp_ms + 1
        return candidate

    @staticmethod
    def _extract(result: Any, *, width: int, height: int) -> list[FingertipDetection]:
        detections: list[FingertipDetection] = []
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            if len(landmarks) <= FINGERTIP_LANDMARK_INDEX:
                continue
            tip = landmarks[FINGERTIP_LANDMARK_INDEX]
            try:
                u_px, v_px = normalized_to_pixel(
                    float(tip.x),
                    float(tip.y),
                    width=width,
                    height=height,
                )
            except ValueError:
                continue

            handedness, handedness_score = HandLandmarker._handedness(result, hand_index)

            detections.append(
                FingertipDetection(
                    hand_index=hand_index,
                    landmark_index=FINGERTIP_LANDMARK_INDEX,
                    landmark_name=FINGERTIP_LANDMARK_NAME,
                    x_normalized=float(tip.x),
                    y_normalized=float(tip.y),
                    u_px=u_px,
                    v_px=v_px,
                    handedness=handedness,
                    handedness_score=handedness_score,
                )
            )
        return detections

    @staticmethod
    def _extract_hands(result: Any, *, width: int, height: int) -> list[HandDetection]:
        detections: list[HandDetection] = []
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            observations: list[LandmarkObservation] = []
            for landmark_index, landmark in enumerate(landmarks[: len(HAND_LANDMARK_NAMES)]):
                x_normalized = float(landmark.x)
                y_normalized = float(landmark.y)
                z_relative = float(landmark.z)
                try:
                    u_px, v_px = normalized_to_pixel(
                        x_normalized,
                        y_normalized,
                        width=width,
                        height=height,
                    )
                except ValueError:
                    u_px = None
                    v_px = None
                    in_frame = False
                else:
                    in_frame = True
                observations.append(
                    LandmarkObservation(
                        landmark_index=landmark_index,
                        landmark_name=HAND_LANDMARK_NAMES[landmark_index],
                        x_normalized=x_normalized,
                        y_normalized=y_normalized,
                        z_mediapipe_relative=z_relative,
                        u_px=u_px,
                        v_px=v_px,
                        in_frame=in_frame,
                    )
                )

            handedness, handedness_score = HandLandmarker._handedness(result, hand_index)
            detections.append(
                HandDetection(
                    hand_index=hand_index,
                    handedness=handedness,
                    handedness_score=handedness_score,
                    landmarks=tuple(observations),
                )
            )
        return detections

    @staticmethod
    def _handedness(result: Any, hand_index: int) -> tuple[str | None, float | None]:
        if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
            return None, None
        category = result.handedness[hand_index][0]
        name = category.category_name or category.display_name or None
        return name, float(category.score)
