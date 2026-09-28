"""Public API for the reproducible video-to-student training pipeline."""

from .config import (
    DatasetConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PrepareConfig,
    SpikeFilterSettings,
    StudentModelSettings,
    StudentOptimizerSettings,
    TeacherConfig,
    TrainingConfig,
    VideoOverrideConfig,
    load_pipeline_config,
)
from .discovery import (
    DiscoveredVideo,
    DuplicateVideoError,
    VideoDiscoveryError,
    discover_videos,
)

__all__ = [
    "DatasetConfig",
    "DiscoveredVideo",
    "DuplicateVideoError",
    "InputConfig",
    "OutputConfig",
    "PipelineConfig",
    "PrepareConfig",
    "SpikeFilterSettings",
    "StudentModelSettings",
    "StudentOptimizerSettings",
    "TeacherConfig",
    "TrainingConfig",
    "VideoDiscoveryError",
    "VideoOverrideConfig",
    "discover_videos",
    "load_pipeline_config",
]
