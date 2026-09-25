from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fingertip_depth.geometry import CameraPoint3D
from fingertip_depth.student_training import StudentTrainingSample
from fingertip_depth.student_trajectory_demo import (
    DemoObservation,
    PredictedObservation,
    _model_config_from_checkpoint,
    _padded_bounds,
    render_demo_frame,
)
from fingertip_depth.trajectory import TrajectoryPoint


def _prediction(frame_index: int, *, x_m: float, z_m: float) -> PredictedObservation:
    sample = StudentTrainingSample(
        sample_id=f"finger_movement_2030:{frame_index:06d}:hand0:aug=identity",
        source_sample_id=f"finger_movement_2030:{frame_index:06d}:hand0",
        split="train",
        image_path=Path("unused.png"),
        image_png_sha256="0" * 64,
        width=640,
        height=480,
        landmark_xy=((0.4, 0.7), (0.45, 0.6), (0.5, 0.5), (0.55, 0.4)),
        target_depth_m=0.25,
        source_sequence_id="finger_movement_2030",
        hand_index=0,
        frame_index=frame_index,
        augmentation_variant="identity",
    )
    observation = DemoObservation(
        sample=sample,
        timestamp_ms=frame_index * 17,
        landmark_pixels=((256, 336), (288, 288), (320, 240), (352, 192)),
    )
    point = TrajectoryPoint.from_camera_point(
        frame_index=frame_index,
        timestamp_ms=frame_index * 17,
        u_px=352.0,
        v_px=192.0,
        point=CameraPoint3D(x_m=x_m, y_m=-0.01, z_m=z_m),
    )
    return PredictedObservation(observation=observation, trajectory_point=point)


def test_checkpoint_model_config_ignores_metadata_and_disables_download() -> None:
    config = _model_config_from_checkpoint(
        {
            "image_encoder_name": "vit_small_patch16_224.dino",
            "pretrained_image_encoder": True,
            "landmark_indices": [5, 6, 7, 8],
            "fusion_layers": 2,
            "fusion_heads": 6,
            "fusion_mlp_ratio": 4.0,
            "dropout": 0.1,
            "landmark_names": ["metadata", "is", "not", "constructor input"],
        }
    )

    assert config.pretrained_image_encoder is False
    assert config.landmark_indices == (5, 6, 7, 8)


def test_padded_bounds_expand_constant_values() -> None:
    low, high = _padded_bounds([0.25, 0.25], minimum_span=0.05)

    assert low < 0.25 < high
    assert high - low == pytest.approx(0.058)


def test_render_demo_frame_combines_source_and_xz_panel() -> None:
    predictions = (
        _prediction(1, x_m=-0.02, z_m=0.20),
        _prediction(2, x_m=0.01, z_m=0.28),
    )
    source = np.zeros((480, 640, 3), dtype=np.uint8)

    rendered = render_demo_frame(
        source,
        frame_index=2,
        current_observation=predictions[-1],
        visible_points=[prediction.trajectory_point for prediction in predictions],
        all_points=[prediction.trajectory_point for prediction in predictions],
        validation_start_frame=9,
        panel_width=420,
    )

    assert rendered.shape == (480, 1060, 3)
    assert rendered.dtype == np.uint8
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(rendered[:, :640]) > 0
    assert np.count_nonzero(rendered[:, 640:]) > 0


def test_demo_script_has_no_sequence_switch() -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "render_student_trajectory_demo.py"
    )
    specification = importlib.util.spec_from_file_location("trajectory_demo_script", script_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args([])
    assert "finger_movement_2030" in str(args.output_dir)
    assert not hasattr(args, "sequence")
