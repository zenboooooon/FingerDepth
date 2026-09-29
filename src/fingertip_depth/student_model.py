'画像特徴を抽出するViTと手ランドマーク特徴を扱うTransformerを組み合わせ、指先の深度を推定するモデルを定義します。'

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import timm
import torch
from torch import nn

from .constants import DEFAULT_FEATURE_LANDMARK_INDICES, HAND_LANDMARK_NAMES

DEFAULT_IMAGE_ENCODER = "vit_small_patch16_224.dino"


# ランドマーク番号が範囲内で重複のない選択か検証します。
def _validate_landmark_indices(indices: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(int(index) for index in indices)
    if not selected:
        raise ValueError("at least one input landmark is required")
    if len(set(selected)) != len(selected):
        raise ValueError("input landmark indices must be unique")
    for index in selected:
        if not 0 <= index < len(HAND_LANDMARK_NAMES):
            raise ValueError(
                f"landmark index must be in [0, {len(HAND_LANDMARK_NAMES) - 1}]: {index}"
            )
    return selected


# 生徒モデルの入力寸法や層構成など、ネットワーク設計の設定です。
@dataclass(frozen=True, slots=True)
class StudentModelConfig:
    """Architecture configuration for the first Phase 8 baseline."""

    image_encoder_name: str = DEFAULT_IMAGE_ENCODER
    pretrained_image_encoder: bool = True
    landmark_indices: tuple[int, ...] = DEFAULT_FEATURE_LANDMARK_INDICES
    fusion_layers: int = 2
    fusion_heads: int = 6
    fusion_mlp_ratio: float = 4.0
    dropout: float = 0.1

    # 作成後にフィールドの型、範囲、相互の整合性を検証します。
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "landmark_indices",
            _validate_landmark_indices(self.landmark_indices),
        )
        if not self.image_encoder_name:
            raise ValueError("image encoder name must not be empty")
        if self.fusion_layers <= 0:
            raise ValueError("fusion_layers must be positive")
        if self.fusion_heads <= 0:
            raise ValueError("fusion_heads must be positive")
        if self.fusion_mlp_ratio <= 0:
            raise ValueError("fusion_mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    # 主要なフィールドを、JSONへ保存できる辞書に変換します。
    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["landmark_indices"] = list(self.landmark_indices)
        value["landmark_names"] = [HAND_LANDMARK_NAMES[index] for index in self.landmark_indices]
        return value


# ランドマーク種別ごとの学習可能なトークン表現を管理します。
class LandmarkTypeTokenBank(nn.Module):
    """One independent token Parameter for every MediaPipe hand landmark.

    A ``ParameterList`` is intentional. Unselected parameters have
    ``requires_grad=False`` and are excluded from the optimizer, so they remain
    bit-for-bit unchanged even when AdamW weight decay is enabled.
    """

    # 必要な引数を検証し、インスタンスの状態を初期化します。
    def __init__(self, *, embedding_dim: int, trainable_indices: Sequence[int]) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.trainable_indices = _validate_landmark_indices(trainable_indices)
        selected = frozenset(self.trainable_indices)
        self.tokens = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_dim)) for _ in HAND_LANDMARK_NAMES]
        )
        for index, token in enumerate(self.tokens):
            nn.init.trunc_normal_(token, std=0.02)
            token.requires_grad_(index in selected)

    # 入力をニューラルネットワークに通し、予測値を返します。
    def forward(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        requested = (
            self.trainable_indices if indices is None else _validate_landmark_indices(indices)
        )
        if requested != self.trainable_indices:
            raise ValueError(
                "runtime landmark order must match the model configuration: "
                f"expected {self.trainable_indices}, got {requested}"
            )
        return torch.stack([self.tokens[index] for index in requested], dim=0)

    # 学習対象のランドマークトークンパラメーターを返します。
    def trainable_token_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.tokens[index] for index in self.trainable_indices)

    # 固定された事前学習トークンパラメーターを返します。
    def frozen_token_parameters(self) -> tuple[nn.Parameter, ...]:
        selected = frozenset(self.trainable_indices)
        return tuple(token for index, token in enumerate(self.tokens) if index not in selected)


# 画像と手ランドマークから指先深度を推定するニューラルネットワークです。
class FingertipDepthStudent(nn.Module):
    """Fuse ViT image tokens and typed hand-landmark tokens for depth regression."""

    # 必要な引数を検証し、インスタンスの状態を初期化します。
    def __init__(
        self,
        config: StudentModelConfig,
        *,
        initial_depth_bias_m: float,
        image_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not torch.isfinite(torch.tensor(initial_depth_bias_m)):
            raise ValueError("initial_depth_bias_m must be finite")
        self.config = config
        self.image_encoder = image_encoder or timm.create_model(
            config.image_encoder_name,
            pretrained=config.pretrained_image_encoder,
            num_classes=0,
            global_pool="",
        )
        embedding_dim = int(getattr(self.image_encoder, "num_features", 0))
        if embedding_dim <= 0:
            raise ValueError("image encoder must expose a positive num_features")
        if embedding_dim % config.fusion_heads != 0:
            raise ValueError(
                f"encoder dimension {embedding_dim} is not divisible by "
                f"fusion_heads={config.fusion_heads}"
            )
        self.embedding_dim = embedding_dim

        # モデル入力に使うのは画像平面上のX・Y座標です。MediaPipeの相対Zは
        # 深度の近道となる情報が漏れる可能性があるため、意図的に除外します。
        self.landmark_coordinate_mlp = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.landmark_type_tokens = LandmarkTypeTokenBank(
            embedding_dim=embedding_dim,
            trainable_indices=config.landmark_indices,
        )
        self.depth_query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.query_modality_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.image_modality_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.landmark_modality_token = nn.Parameter(torch.empty(1, 1, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=config.fusion_heads,
            dim_feedforward=round(embedding_dim * config.fusion_mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.fusion_layers,
            norm=nn.LayerNorm(embedding_dim),
            enable_nested_tensor=False,
        )
        hidden_dim = max(embedding_dim // 2, 1)
        self.depth_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_new_parameters(initial_depth_bias_m=float(initial_depth_bias_m))

    # 事前学習重みのない新規ネットワーク層を初期化します。
    def _initialize_new_parameters(self, *, initial_depth_bias_m: float) -> None:
        modules = (
            self.landmark_coordinate_mlp,
            self.fusion_transformer,
            self.depth_head,
        )
        for root in modules:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        for parameter in (
            self.depth_query,
            self.query_modality_token,
            self.image_modality_token,
            self.landmark_modality_token,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        final = self.depth_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, initial_depth_bias_m)

    # 画像をViTで符号化し、画像特徴トークンを返します。
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        features = self.image_encoder.forward_features(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise TypeError("image encoder forward_features must return [batch, tokens, channels]")
        if features.shape[0] != images.shape[0] or features.shape[2] != self.embedding_dim:
            raise ValueError("image encoder token shape is incompatible with the fusion model")
        return features

    # 画像トークンとランドマークトークンを統合して深度を回帰します。
    def forward_from_tokens(
        self,
        image_tokens: torch.Tensor,
        landmark_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        expected_landmarks = len(self.config.landmark_indices)
        if image_tokens.ndim != 3 or image_tokens.shape[2] != self.embedding_dim:
            raise ValueError("image_tokens must have shape [batch, tokens, embedding_dim]")
        if landmark_coordinates.ndim != 3 or landmark_coordinates.shape[2] != 2:
            raise ValueError("landmark_coordinates must have shape [batch, landmarks, 2]")
        if landmark_coordinates.shape[:2] != (
            image_tokens.shape[0],
            expected_landmarks,
        ):
            raise ValueError(
                "landmark batch/count differs from image tokens or model configuration"
            )

        batch_size = image_tokens.shape[0]
        image_tokens = image_tokens + self.image_modality_token
        coordinate_tokens = self.landmark_coordinate_mlp(landmark_coordinates)
        type_tokens = self.landmark_type_tokens().unsqueeze(0)
        landmark_tokens = coordinate_tokens + type_tokens + self.landmark_modality_token
        depth_query = self.depth_query.expand(batch_size, -1, -1) + self.query_modality_token
        fused = torch.cat((depth_query, image_tokens, landmark_tokens), dim=1)
        encoded = self.fusion_transformer(fused)
        depth_logits_m = self.depth_head(encoded[:, 0]).squeeze(-1)
        if depth_logits_m.shape != (batch_size,):
            raise AssertionError("depth head returned an unexpected shape")
        return depth_logits_m

    # 入力をニューラルネットワークに通し、予測値を返します。
    def forward(
        self,
        images: torch.Tensor,
        landmark_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_from_tokens(self.encode_images(images), landmark_coordinates)

    # モデル全体の学習可能・固定パラメーター数を集計します。
    def parameter_counts(self) -> dict[str, int]:
        all_parameters = tuple(self.parameters())
        encoder_parameters = tuple(self.image_encoder.parameters())
        type_parameters = tuple(self.landmark_type_tokens.tokens)
        return {
            "total": sum(parameter.numel() for parameter in all_parameters),
            "trainable": sum(
                parameter.numel() for parameter in all_parameters if parameter.requires_grad
            ),
            "image_encoder_total": sum(parameter.numel() for parameter in encoder_parameters),
            "image_encoder_trainable": sum(
                parameter.numel() for parameter in encoder_parameters if parameter.requires_grad
            ),
            "landmark_type_tokens_total": sum(parameter.numel() for parameter in type_parameters),
            "landmark_type_tokens_trainable": sum(
                parameter.numel() for parameter in type_parameters if parameter.requires_grad
            ),
        }


__all__ = [
    "DEFAULT_IMAGE_ENCODER",
    "FingertipDepthStudent",
    "LandmarkTypeTokenBank",
    "StudentModelConfig",
]
