from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import pytest

from fingertip_depth import pseudo_label_inputs
from fingertip_depth.constants import HAND_LANDMARK_NAMES
from fingertip_depth.hands import HandDetection, LandmarkObservation
from fingertip_depth.pseudo_label_inputs import (
    iter_prepared_records,
    prepare_pseudo_label_inputs,
)
from fingertip_depth.video_cache import (
    pixel_hash_sequence_sha256,
    pixel_sha256,
    sha256_file,
)


def _hand(*, out_of_frame_index: int | None = None) -> HandDetection:
    landmarks = []
    for index in (5, 6, 7, 8):
        in_frame = index != out_of_frame_index
        landmarks.append(
            LandmarkObservation(
                landmark_index=index,
                landmark_name=HAND_LANDMARK_NAMES[index],
                x_normalized=0.2 + index / 100.0,
                y_normalized=0.3 + index / 100.0,
                z_mediapipe_relative=-index / 1000.0,
                u_px=10 + index if in_frame else None,
                v_px=20 + index if in_frame else None,
                in_frame=in_frame,
            )
        )
    return HandDetection(
        hand_index=0,
        handedness="Right",
        handedness_score=0.95,
        landmarks=tuple(landmarks),
    )


def _install_fake_landmarker(
    monkeypatch: pytest.MonkeyPatch,
    results: list[list[HandDetection]],
) -> list[int]:
    timestamps: list[int] = []

    class FakeHandLandmarker:
        def __init__(self, **_kwargs: object) -> None:
            self._results = iter(results)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def detect_hands(
            self,
            rgb: np.ndarray,
            *,
            timestamp_ms: int | None = None,
        ) -> list[HandDetection]:
            assert rgb.dtype == np.uint8
            assert timestamp_ms is not None
            timestamps.append(timestamp_ms)
            return next(self._results)

    monkeypatch.setattr(pseudo_label_inputs, "HandLandmarker", FakeHandLandmarker)
    return timestamps


def _frame_cache(tmp_path: Path) -> tuple[Path, Path, str, list[np.ndarray]]:
    source_video = tmp_path / "source.mov"
    source_video.write_bytes(b"fixed source video")
    cache_dir = tmp_path / "cache"
    frames_dir = cache_dir / "frames"
    frames_dir.mkdir(parents=True)
    frames = [
        np.full((40, 60, 3), 20, dtype=np.uint8),
        np.full((40, 60, 3), 80, dtype=np.uint8),
    ]
    entries = []
    for index, bgr in enumerate(frames):
        path = frames_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(path), bgr)
        entries.append(
            {
                "frame_index": index,
                "timestamp_ms": index * 40,
                "relative_path": f"frames/frame_{index:06d}.png",
                "width": bgr.shape[1],
                "height": bgr.shape[0],
                "png_sha256": sha256_file(path),
                "bgr_pixel_sha256": pixel_sha256(bgr),
            }
        )
    manifest = {
        "format": "fingertip-depth-lossless-video-frame-cache",
        "format_version": 1,
        "source": str(source_video.resolve()),
        "source_sha256": sha256_file(source_video),
        "baseline_records_sha256": "a" * 64,
        "frame_count": len(entries),
        "fps": 25.0,
        "source_decoder": {"opencv_version": "fixture"},
        "frames": entries,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_video, manifest_path, sha256_file(manifest_path), frames


def test_prepares_audited_cache_with_selected_landmarks_and_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, cache_manifest, cache_sha256, frames = _frame_cache(tmp_path)
    timestamps = _install_fake_landmarker(monkeypatch, [[_hand()], []])
    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"fixed hand model")
    output_dir = tmp_path / "prepared"

    manifest = prepare_pseudo_label_inputs(
        output_dir=output_dir,
        hand_model_path=hand_model,
        frame_cache_manifest_path=cache_manifest,
        expected_frame_cache_manifest_sha256=cache_sha256,
        focal_35mm_equivalent_mm=36.0,
    )

    assert manifest["format"] == "fingertip-depth-pseudo-label-inputs"
    assert manifest["format_version"] == 1
    assert manifest["source"]["mode"] == "audited_frame_cache"
    assert manifest["frame_count"] == 2
    assert manifest["accepted_frame_count"] == 1
    assert manifest["rejected_frame_count"] == 1
    assert manifest["no_hand_frame_count"] == 1
    assert manifest["feature_landmarks"]["indices"] == [5, 6, 7, 8]
    assert manifest["target_landmark"]["index"] == 8
    expected_focal = 36.0 * math.hypot(60, 40) / math.hypot(36.0, 24.0)
    assert manifest["camera"]["intrinsics"]["fx_px"] == pytest.approx(expected_focal)
    assert manifest["camera"]["calibrated"] is False
    assert manifest["frames"]["pixel_sequence_sha256"] == pixel_hash_sequence_sha256(
        pixel_sha256(frame) for frame in frames
    )
    assert timestamps == [0, 40]

    records = list(iter_prepared_records(output_dir / "manifest.json"))
    assert [record["status"] for record in records] == ["accepted", "rejected"]
    assert records[1]["rejection_reasons"] == ["no_hand"]
    assert records[0]["selected_hand_index"] == 0
    assert records[0]["target_landmark"]["landmark_index"] == 8
    assert [landmark["landmark_index"] for landmark in records[0]["hands"][0]["landmarks"]] == [
        5,
        6,
        7,
        8,
    ]
    for index, expected in enumerate(frames):
        stored = cv2.imread(
            str(output_dir / records[index]["image_path"]),
            cv2.IMREAD_COLOR,
        )
        assert np.array_equal(stored, expected)
        assert records[index]["bgr_pixel_sha256"] == pixel_sha256(expected)


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = iter(frames)
        self.released = False
        self.orientation_auto = False

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: float) -> bool:
        if property_id == getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", -1):
            self.orientation_auto = value >= 0.5
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return 30.0
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return 0.0
        if property_id == getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", -1):
            return 1.0 if self.orientation_auto else 0.0
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self._frames).copy()
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


def test_direct_video_mode_writes_every_frame_and_strict_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "input.mov"
    video.write_bytes(b"video identity")
    frames = [
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.full((4, 6, 3), 12, dtype=np.uint8),
    ]
    capture = _FakeCapture(frames)
    monkeypatch.setattr(pseudo_label_inputs.cv2, "VideoCapture", lambda _path: capture)
    timestamps = _install_fake_landmarker(monkeypatch, [[], []])
    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"hand")

    manifest = prepare_pseudo_label_inputs(
        output_dir=tmp_path / "out",
        hand_model_path=hand_model,
        input_video_path=video,
    )

    assert capture.released is True
    assert timestamps == [0, 33]
    assert manifest["source"]["mode"] == "direct_video_decode"
    assert manifest["frame_count"] == 2
    assert manifest["accepted_frame_count"] == 0
    assert manifest["no_hand_frame_count"] == 2


def test_rejects_out_of_frame_feature_and_detects_records_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, cache_manifest, cache_sha256, _frames = _frame_cache(tmp_path)
    _install_fake_landmarker(monkeypatch, [[_hand(out_of_frame_index=6)]])
    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"hand")
    output_dir = tmp_path / "prepared"

    manifest = prepare_pseudo_label_inputs(
        output_dir=output_dir,
        hand_model_path=hand_model,
        frame_cache_manifest_path=cache_manifest,
        expected_frame_cache_manifest_sha256=cache_sha256,
        max_frames=1,
    )

    assert manifest["accepted_frame_count"] == 0
    record = next(iter_prepared_records(output_dir / "manifest.json"))
    assert record["rejection_reasons"] == ["feature_landmark_out_of_frame:6"]
    with (output_dir / "frames.jsonl").open("a", encoding="utf-8") as target:
        target.write("{}\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        list(iter_prepared_records(output_dir / "manifest.json"))


def test_requires_exactly_one_source_mode(tmp_path: Path) -> None:
    hand_model = tmp_path / "hand.task"
    hand_model.write_bytes(b"hand")

    with pytest.raises(ValueError, match="exactly one"):
        prepare_pseudo_label_inputs(
            output_dir=tmp_path / "out",
            hand_model_path=hand_model,
        )
