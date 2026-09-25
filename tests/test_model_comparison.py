import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fingertip_depth import model_comparison
from fingertip_depth.camera import CameraIntrinsics


class _FakeEstimator:
    def __init__(self, depths: list[float]) -> None:
        self.metadata = {"checkpoint_sha256": "abc", "test_double": True}
        self.depths = iter(depths)
        self.calls: list[tuple[CameraIntrinsics | None, str]] = []

    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> SimpleNamespace:
        self.calls.append((intrinsics, camera_mode))
        depth = np.full(rgb.shape[:2], next(self.depths), dtype=np.float32)
        return SimpleNamespace(
            depth_m=depth,
            inference_ms=3.0,
            device="test",
            extras={"predicted_intrinsics": {"fx_px": 123.0}},
        )


def _green_box_image(path: Path) -> None:
    bgr = np.full((120, 160, 3), 180, dtype=np.uint8)
    green = cv2.cvtColor(np.uint8([[[70, 180, 120]]]), cv2.COLOR_HSV2BGR)[0, 0]
    bgr[60:110, 55:105] = green
    assert cv2.imwrite(str(path), bgr)


def test_known_distance_condition_passes_approximate_k_without_rescaling(
    tmp_path: Path,
) -> None:
    first = tmp_path / "05image.png"
    second = tmp_path / "10image.png"
    _green_box_image(first)
    _green_box_image(second)
    estimator = _FakeEstimator([0.6, 1.2])
    condition = model_comparison.comparison_condition("unidepth_v2_l__approx_k")

    summary = model_comparison.evaluate_known_distance_condition(
        condition=condition,
        estimator=estimator,
        samples=[(first, 0.5), (second, 1.0)],
        output_dir=tmp_path / "out",
    )

    assert [row["predicted_depth_m"] for row in summary["rows"]] == pytest.approx(
        [0.6, 1.2]
    )
    assert summary["aggregate"]["scale_fit_applied_to_reported_predictions"] is False
    assert summary["inference_ms"]["count"] == 2
    assert summary["prediction_extras_numeric_summary"][
        "predicted_intrinsics.fx_px"
    ]["median"] == pytest.approx(123.0)
    assert all(isinstance(call[0], CameraIntrinsics) for call in estimator.calls)
    assert [call[1] for call in estimator.calls] == ["approx_k", "approx_k"]
    assert (tmp_path / "out" / "05image" / "depth_m.npy").is_file()


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return 30.0 if property_id == cv2.CAP_PROP_FPS else 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self.frames).copy()
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


def _baseline_records(path: Path) -> str:
    records = [
        {
            "source": "input.mov",
            "frame_index": 0,
            "timestamp_ms": 0,
            "width": 5,
            "height": 4,
            "hand_detected": False,
            "fingertips": [],
        },
        {
            "source": "input.mov",
            "frame_index": 1,
            "timestamp_ms": 33,
            "width": 5,
            "height": 4,
            "hand_detected": True,
            "fingertips": [
                {
                    "hand_index": 0,
                    "landmark_index": 8,
                    "landmark_name": "INDEX_FINGER_TIP",
                    "x_normalized": 0.5,
                    "y_normalized": 0.5,
                    "u_px": 2,
                    "v_px": 1,
                    "handedness": "Right",
                    "handedness_score": 0.9,
                    "depth_m": 999.0,
                    "depth_valid": True,
                }
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_video_condition_reuses_verified_coordinates_without_mediapipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_sha256 = _baseline_records(baseline_path)
    video_path = tmp_path / "input.mov"
    video_path.write_bytes(b"test video identity")
    capture = _FakeCapture(
        [np.full((4, 5, 3), value, dtype=np.uint8) for value in (10, 20)]
    )
    monkeypatch.setattr(model_comparison.cv2, "VideoCapture", lambda _path: capture)
    estimator = _FakeEstimator([1.0, 2.5])
    condition = model_comparison.comparison_condition("depth_pro__estimated_focal")

    summary = model_comparison.evaluate_video_condition(
        condition=condition,
        estimator=estimator,
        input_path=video_path,
        baseline_records_path=baseline_path,
        baseline_records_sha256=baseline_sha256,
        output_dir=tmp_path / "out",
        write_annotated_video=False,
    )

    output_records = [
        json.loads(line)
        for line in (tmp_path / "out" / "frames.jsonl").read_text().splitlines()
    ]
    assert output_records[0]["fingertips"] == []
    assert output_records[1]["fingertips"][0]["u_px"] == 2
    assert output_records[1]["fingertips"][0]["v_px"] == 1
    assert output_records[1]["fingertips"][0]["depth_m"] == pytest.approx(2.5)
    assert summary["baseline_coordinate_source"]["verified_sha256"] == baseline_sha256
    assert summary["baseline_coordinate_source"]["media_pipe_rerun"] is False
    assert summary["processed_frames"] == 2
    assert summary["frames_with_valid_fingertip_depth"] == 1
    assert all(intrinsics is None for intrinsics, _mode in estimator.calls)
    assert capture.released is True
    assert (tmp_path / "out" / "fingertip_depth_timeseries.png").is_file()


def test_video_condition_rejects_unexpected_baseline_hash(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    _baseline_records(baseline_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model_comparison.evaluate_video_condition(
            condition=model_comparison.comparison_condition(
                "unidepth_v2_l__no_camera"
            ),
            estimator=_FakeEstimator([1.0]),
            input_path=tmp_path / "input.mov",
            baseline_records_path=baseline_path,
            baseline_records_sha256="0" * 64,
            output_dir=tmp_path / "out",
        )
