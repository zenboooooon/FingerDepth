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
