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
