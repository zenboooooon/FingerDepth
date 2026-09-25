from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from fingertip_depth.coreml_export import (
    StudentCoreMLWrapper,
    _contained_artifact,
    trace_student_for_coreml,
)


class _RecordingStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.images: torch.Tensor | None = None
        self.landmarks: torch.Tensor | None = None

    def forward(self, images: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        self.images = images
        self.landmarks = landmarks
        return images.mean(dim=(1, 2, 3)) + landmarks.mean(dim=(1, 2))


def test_coreml_wrapper_owns_exact_training_transforms() -> None:
    student = _RecordingStudent()
    wrapper = StudentCoreMLWrapper(
        student,  # type: ignore[arg-type]
        image_mean=(0.5, 0.25, 0.75),
        image_std=(0.25, 0.5, 0.125),
    )
    image = torch.tensor([0.5, 0.25, 0.75]).view(1, 3, 1, 1)
    landmarks = torch.tensor([[[0.0, 0.25], [0.5, 0.75], [1.0, 0.5], [0.2, 0.8]]])

    output = wrapper(image, landmarks)

    assert output.shape == (1, 1)
    assert student.images is not None
    assert student.landmarks is not None
    torch.testing.assert_close(student.images, torch.zeros_like(student.images))
    torch.testing.assert_close(student.landmarks, landmarks * 2.0 - 1.0)


def test_coreml_wrapper_rejects_invalid_normalization() -> None:
    with pytest.raises(ValueError, match="three"):
        StudentCoreMLWrapper(
            _RecordingStudent(),  # type: ignore[arg-type]
            image_mean=(0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="positive"):
        StudentCoreMLWrapper(
            _RecordingStudent(),  # type: ignore[arg-type]
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 0.0, 1.0),
        )


def test_contained_artifact_rejects_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.pt"
    outside.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="escapes"):
        _contained_artifact(tmp_path, "../outside.pt", field="checkpoint")


def test_trace_student_wrapper_round_trips(tmp_path: Path) -> None:
    wrapper = StudentCoreMLWrapper(
        _RecordingStudent(),  # type: ignore[arg-type]
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ).eval()
    output_path = tmp_path / "student.trace.pt"

    traced, metrics = trace_student_for_coreml(wrapper, output_path=output_path)

    assert output_path.is_file()
    assert metrics["export_wrapper_vs_torchscript_max_absolute_difference_m"] == pytest.approx(0.0)
    image = torch.full((1, 3, 224, 224), 0.5)
    landmarks = torch.full((1, 4, 2), 0.5)
    torch.testing.assert_close(traced(image, landmarks), wrapper(image, landmarks))
