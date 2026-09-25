from __future__ import annotations

import torch
from torch import nn

from fingertip_depth.coreml_export import CoreMLTraceWrapper, StudentCoreMLWrapper
from fingertip_depth.student_model import FingertipDepthStudent, StudentModelConfig


class _FakePatchEmbed(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        value = images.mean(dim=(1, 2, 3)).reshape(batch, 1, 1)
        return value.expand(batch, 196, 384)


class _FakeViT(nn.Module):
    num_features = 384

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _FakePatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 384) * 0.01)
        self.pos_drop = nn.Identity()
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.Sequential(nn.LayerNorm(384))
        self.norm = nn.Identity()

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        tokens = self.blocks(tokens)
        return self.norm(tokens)


def test_coreml_trace_wrapper_matches_eval_transformer() -> None:
    torch.manual_seed(7)
    config = StudentModelConfig(
        pretrained_image_encoder=False,
        landmark_indices=(5, 6, 7, 8),
        fusion_layers=1,
        fusion_heads=6,
        dropout=0.0,
    )
    model = FingertipDepthStudent(
        config,
        initial_depth_bias_m=0.25,
        image_encoder=_FakeViT(),
    ).eval()
    final = model.depth_head[-1]
    assert isinstance(final, nn.Linear)
    nn.init.normal_(final.weight, std=0.01)

    direct = StudentCoreMLWrapper(
        model,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ).eval()
    compatible = CoreMLTraceWrapper(
        model,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ).eval()
    images = torch.rand(1, 3, 224, 224)
    landmarks = torch.rand(1, 4, 2)

    with torch.inference_mode():
        expected = direct(images, landmarks)
        observed = compatible(images, landmarks)

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
