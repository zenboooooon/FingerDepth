from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fingertip_depth.coreml_export import (
    StudentCoreMLWrapper,
    _contained_artifact,
    export_student_for_coreml,
    rebuild_parity_fixture,
    sha256_file,
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


class _PureStudent(nn.Module):
    def forward(self, images: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
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


def test_export_student_wrapper_round_trips_as_aten_pt2(tmp_path: Path) -> None:
    wrapper = StudentCoreMLWrapper(
        _PureStudent(),  # type: ignore[arg-type]
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ).eval()
    output_path = tmp_path / "StudentDepthLatest.pt2"

    exported, metrics = export_student_for_coreml(wrapper, output_path=output_path)

    assert output_path.is_file()
    assert exported.dialect == "ATEN"
    assert metrics["strict"] is True
    assert metrics["decomposition_table"] == "empty"
    assert metrics["forbidden_operators"] == []
    assert (
        metrics["trained_pytorch_vs_exported_program_max_absolute_difference_m"]
        == pytest.approx(0.0)
    )
    loaded = torch.export.load(output_path)
    image = torch.full((1, 3, 224, 224), 0.5)
    landmarks = torch.full((1, 4, 2), 0.5)
    torch.testing.assert_close(loaded.module()(image, landmarks), wrapper(image, landmarks))


def test_export_student_requires_pt2_suffix(tmp_path: Path) -> None:
    wrapper = StudentCoreMLWrapper(
        _PureStudent(),  # type: ignore[arg-type]
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ).eval()

    with pytest.raises(ValueError, match=".pt2"):
        export_student_for_coreml(wrapper, output_path=tmp_path / "model.pt")


def test_rebuild_parity_fixture_preserves_inputs_and_recomputes_predictions(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "legacy_inputs.npz"
    images = np.stack(
        [np.full((224, 224, 3), value, dtype=np.uint8) for value in (64, 128, 192)]
    )
    landmarks = np.asarray(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
            [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        source_path,
        images_rgb_uint8=images,
        landmarks_xy=landmarks,
        pytorch_depth_m=np.zeros(3, dtype=np.float32),
        frame_index=np.asarray([10, 11, 12], dtype=np.int32),
        sample_id=np.asarray(["a", "b", "c"]),
    )
    wrapper = StudentCoreMLWrapper(
        _PureStudent(),  # type: ignore[arg-type]
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ).eval()
    output_path = tmp_path / "latest_inputs.npz"

    summary = rebuild_parity_fixture(
        wrapper,
        source_path,
        output_path=output_path,
        batch_size=1,
        limit=2,
    )

    assert summary["sample_count"] == 2
    assert summary["input_source_fixture_sha256"] == sha256_file(source_path)
    with np.load(output_path, allow_pickle=False) as rebuilt:
        np.testing.assert_array_equal(rebuilt["images_rgb_uint8"], images[:2])
        np.testing.assert_array_equal(rebuilt["landmarks_xy"], landmarks[:2])
        np.testing.assert_array_equal(rebuilt["frame_index"], [10, 11])
        np.testing.assert_array_equal(rebuilt["sample_id"], ["a", "b"])
        assert not np.array_equal(
            rebuilt["pytorch_depth_m"],
            np.zeros(2, dtype=np.float32),
        )
