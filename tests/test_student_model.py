from __future__ import annotations

import pytest
import torch
from torch import nn

from fingertip_depth.constants import HAND_LANDMARK_NAMES
from fingertip_depth.student_model import FingertipDepthStudent, StudentModelConfig


class TinyImageEncoder(nn.Module):
    """Small ViT-shaped test double that never needs an external checkpoint."""

    def __init__(self, embedding_dim: int = 12) -> None:
        super().__init__()
        self.num_features = embedding_dim
        self.projection = nn.Conv2d(3, embedding_dim, kernel_size=2, stride=2)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.projection(images)
        return features.flatten(2).transpose(1, 2)


def _model(*, initial_depth_bias_m: float = 0.3) -> FingertipDepthStudent:
    config = StudentModelConfig(
        image_encoder_name="fake-tiny-encoder",
        pretrained_image_encoder=False,
        landmark_indices=(5, 6, 7, 8),
        fusion_layers=1,
        fusion_heads=3,
        fusion_mlp_ratio=2.0,
        dropout=0.0,
    )
    return FingertipDepthStudent(
        config,
        initial_depth_bias_m=initial_depth_bias_m,
        image_encoder=TinyImageEncoder(),
    )


def _valid_inputs(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(batch_size, 3, 4, 4)
    coordinates = torch.rand(batch_size, 4, 2)
    return images, coordinates


@pytest.mark.parametrize("initial_bias", [0.2772695817, -0.25])
def test_output_shape_and_initial_prediction_are_raw_head_bias(initial_bias: float) -> None:
    torch.manual_seed(0)
    model = _model(initial_depth_bias_m=initial_bias).eval()
    images, coordinates = _valid_inputs(batch_size=3)

    with torch.no_grad():
        prediction = model(images, coordinates)

    assert prediction.shape == (3,)
    torch.testing.assert_close(
        prediction,
        torch.full((3,), initial_bias, dtype=prediction.dtype),
        rtol=0.0,
        atol=0.0,
    )


def test_all_21_type_tokens_exist_but_only_selected_tokens_are_trainable() -> None:
    model = _model()
    bank = model.landmark_type_tokens

    assert len(HAND_LANDMARK_NAMES) == 21
    assert len(bank.tokens) == 21
    assert all(token.shape == (model.embedding_dim,) for token in bank.tokens)
    assert bank.trainable_indices == (5, 6, 7, 8)
    assert {index for index, token in enumerate(bank.tokens) if token.requires_grad} == {
        5,
        6,
        7,
        8,
    }
    assert len(bank.trainable_token_parameters()) == 4
    assert len(bank.frozen_token_parameters()) == 17


def test_only_selected_type_tokens_receive_gradients_and_adamw_updates() -> None:
    torch.manual_seed(1)
    model = _model()
    images, coordinates = _valid_inputs()
    tokens_before = [token.detach().clone() for token in model.landmark_type_tokens.tokens]

    final_layer = model.depth_head[-1]
    assert isinstance(final_layer, nn.Linear)
    with torch.no_grad():
        final_layer.weight.copy_(
            torch.linspace(
                0.01,
                0.05,
                steps=final_layer.weight.numel(),
                dtype=final_layer.weight.dtype,
            ).reshape_as(final_layer.weight)
        )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
        weight_decay=0.1,
    )
    optimizer.zero_grad(set_to_none=True)
    model(images, coordinates).square().mean().backward()

    for index, token in enumerate(model.landmark_type_tokens.tokens):
        if index in {5, 6, 7, 8}:
            assert token.grad is not None
            assert torch.isfinite(token.grad).all()
            assert torch.count_nonzero(token.grad) > 0
        else:
            assert token.grad is None

    optimizer.step()

    for index, (before, after) in enumerate(
        zip(tokens_before, model.landmark_type_tokens.tokens, strict=True)
    ):
        if index in {5, 6, 7, 8}:
            assert not torch.equal(before, after.detach())
        else:
            assert torch.equal(before, after.detach())


@pytest.mark.parametrize(
    "indices",
    [
        (),
        (5, 5),
        (-1,),
        (21,),
    ],
)
def test_config_rejects_invalid_landmark_indices(indices: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        StudentModelConfig(
            image_encoder_name="fake-tiny-encoder",
            pretrained_image_encoder=False,
            landmark_indices=indices,
        )


def test_runtime_type_token_order_must_match_configured_landmark_order() -> None:
    bank = _model().landmark_type_tokens

    with pytest.raises(ValueError, match="runtime landmark order"):
        bank((8, 7, 6, 5))


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(2, 4, 4),
        torch.zeros(2, 1, 4, 4),
        torch.zeros(2, 4, 4, 3),
    ],
)
def test_rejects_invalid_image_shapes(images: torch.Tensor) -> None:
    model = _model()

    with pytest.raises(ValueError, match=r"\[batch, 3, height, width\]"):
        model.encode_images(images)


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        (torch.zeros(2, 4), r"\[batch, landmarks, 2\]"),
        (torch.zeros(2, 4, 1), r"\[batch, landmarks, 2\]"),
        # MediaPipe relative-z is intentionally excluded from the student input.
        (torch.zeros(2, 4, 3), r"\[batch, landmarks, 2\]"),
        (torch.zeros(1, 4, 2), "landmark batch/count"),
        (torch.zeros(2, 3, 2), "landmark batch/count"),
        (torch.zeros(2, 5, 2), "landmark batch/count"),
    ],
)
def test_rejects_invalid_landmark_coordinate_shapes(
    coordinates: torch.Tensor,
    message: str,
) -> None:
    model = _model()
    image_tokens = torch.zeros(2, 4, model.embedding_dim)

    with pytest.raises(ValueError, match=message):
        model.forward_from_tokens(image_tokens, coordinates)


@pytest.mark.parametrize(
    "image_tokens",
    [
        torch.zeros(2, 12),
        torch.zeros(2, 4, 11),
        torch.zeros(2, 4, 13),
    ],
)
def test_rejects_invalid_image_token_shapes(image_tokens: torch.Tensor) -> None:
    model = _model()
    coordinates = torch.zeros(2, 4, 2)

    with pytest.raises(ValueError, match=r"\[batch, tokens, embedding_dim\]"):
        model.forward_from_tokens(image_tokens, coordinates)
