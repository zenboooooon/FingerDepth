from __future__ import annotations

from pathlib import Path

import pytest

from fingertip_depth.student_model import StudentModelConfig
from fingertip_depth.student_training import StudentTrainingConfig, TeacherSpikeFilterConfig
from fingertip_depth.training_pipeline import TeacherConfig, load_pipeline_config


def _write_config(project_root: Path, text: str) -> Path:
    config_path = project_root / "configs" / "nested" / "pipeline.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_load_config_resolves_paths_from_project_root_and_converts_training_types(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
[input]
train_dir = "data/training_videos/train"
validation_dir = "data/training_videos/validation"
extensions = ["MOV", ".mp4", ".m4v"]
default_focal_35mm_mm = 36.0

[output]
processed_dir = "outputs/training_pipeline/sequences"
dataset_dir = "outputs/training_pipeline/datasets"
runs_dir = "outputs/training_pipeline/runs"

[video_overrides."data/training_videos/train/special.MOV"]
focal_35mm_mm = 26.0

[prepare]
hand_model_path = "assets/hand_landmarker.task"
landmark_indices = [5, 6, 7, 8]
frame_transfer_mode = "hardlink"

[teacher]
depth_pro_project = "environments/depth_pro"
device = "cuda:0"
frame_transfer_mode = "hardlink"
checkpoint_interval_frames = 64

[dataset]
frame_transfer_mode = "hardlink"

[training]
device = "cuda:0"

[training.model]
image_encoder_name = "vit_small_patch16_224.dino"
pretrained_image_encoder = false
landmark_indices = [5, 6, 7, 8]
fusion_layers = 3
fusion_heads = 6
fusion_mlp_ratio = 3.0
dropout = 0.2

[training.optimizer]
epochs = 4
batch_size = 2
encoder_learning_rate = 0.00001
head_learning_rate = 0.0001
weight_decay = 0.05
warmup_fraction = 0.1
gradient_clip_norm = 1.0
early_stopping_patience = 2
image_size = 224
image_mean = [0.1, 0.2, 0.3]
image_std = [0.4, 0.5, 0.6]
num_workers = 0
seed = 17
precision = "float32"
freeze_image_encoder = true
preload_images = false
verify_image_png_sha256 = true

[training.spike_filter]
enabled = true
frame_radius = 4
max_frame_gap = 2
min_neighbors = 3
absolute_floor_m = 0.1
relative_floor_fraction = 0.4
mad_multiplier = 5.0
mad_scale = 1.4826
""",
    )

    config = load_pipeline_config(config_path, project_root=tmp_path)

    assert config.config_path == config_path.resolve()
    assert config.input.train_dir == (tmp_path / "data/training_videos/train").resolve()
    assert config.input.extensions == (".m4v", ".mov", ".mp4")
    override_path = (tmp_path / "data/training_videos/train/special.MOV").resolve()
    assert config.input.video_overrides[override_path].focal_35mm_mm == 26.0
    assert config.prepare.hand_model_path == (tmp_path / "assets/hand_landmarker.task").resolve()
    assert config.teacher.depth_pro_project == (tmp_path / "environments/depth_pro").resolve()
    assert config.teacher.checkpoint_interval_frames == 64

    model = config.training.model_config()
    optimizer = config.training.training_config()
    spike_filter = config.training.spike_filter_config()
    assert isinstance(model, StudentModelConfig)
    assert isinstance(optimizer, StudentTrainingConfig)
    assert isinstance(spike_filter, TeacherSpikeFilterConfig)
    assert model.pretrained_image_encoder is False
    assert model.fusion_layers == 3
    assert optimizer.epochs == 4
    assert optimizer.image_mean == pytest.approx((0.1, 0.2, 0.3))
    assert spike_filter.enabled is True
    assert spike_filter.frame_radius == 4


def test_fingerprint_dict_uses_project_relative_paths_and_plain_values(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "")

    config = load_pipeline_config(config_path, project_root=tmp_path)
    payload = config.fingerprint_dict()

    assert payload["input"]["train_dir"] == "data/training_videos/train"
    assert payload["output"]["processed_dir"] == "outputs/training_pipeline/processed"
    assert payload["prepare"]["hand_model_path"] == "assets/hand_landmarker.task"
    assert payload["teacher"]["depth_pro_project"] == "environments/depth_pro"
    assert payload["teacher"]["checkpoint_interval_frames"] == 100
    assert payload["training"]["model"]["landmark_indices"] == [5, 6, 7, 8]
    assert payload["training"]["spike_filter"]["enabled"] is True


@pytest.mark.parametrize("value", (True, 1.5))
def test_teacher_config_rejects_non_integer_checkpoint_interval(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(TypeError, match="checkpoint_interval_frames must be an integer"):
        TeacherConfig(
            depth_pro_project=tmp_path,
            checkpoint_interval_frames=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[input]\ndefault_focal_35mm_mm = 0\n", "must be positive"),
        ('[input]\nextensions = [".avi"]\n', "unsupported input video extension"),
        ("[training.model]\nunknown = 1\n", "unknown training.model"),
        (
            "[teacher]\ncheckpoint_interval_frames = 0\n",
            "teacher.checkpoint_interval_frames must be positive",
        ),
    ],
)
def test_load_config_rejects_invalid_values(tmp_path: Path, text: str, message: str) -> None:
    config_path = _write_config(tmp_path, text)

    with pytest.raises((TypeError, ValueError), match=message):
        load_pipeline_config(config_path, project_root=tmp_path)


def test_load_config_rejects_overlapping_output_and_input_paths(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[output]
processed_dir = "data/training_videos/train/generated"
dataset_dir = "outputs/datasets"
runs_dir = "outputs/runs"
""",
    )

    with pytest.raises(ValueError, match="must not be inside an input directory"):
        load_pipeline_config(config_path, project_root=tmp_path)


def test_load_config_rejects_training_landmarks_not_prepared(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[prepare]
landmark_indices = [5, 6, 7]

[training.model]
landmark_indices = [5, 6, 7, 8]
""",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"training\.model\.landmark_indices must be included in "
            r"prepare\.landmark_indices: missing \[8\]"
        ),
    ):
        load_pipeline_config(config_path, project_root=tmp_path)
