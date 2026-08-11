from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class DepthFusionConfig:
    rgb_feature_dim: int = 4096
    depth_feature_dim: int = 1024
    hidden_dim: int = 2048
    action_dim: int = 7
    dropout: float = 0.1
    use_layer_norm: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class DepthFusionPolicy(nn.Module):
    """Fusion action head for precomputed LAPA RGB and depth features."""

    def __init__(self, config: DepthFusionConfig):
        super().__init__()
        self.config = config
        input_dim = config.rgb_feature_dim + config.depth_feature_dim

        layers = []
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.extend(
            [
                nn.Linear(input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.action_dim),
            ]
        )
        self.action_head = nn.Sequential(*layers)

    def forward(self, rgb_feature: torch.Tensor, depth_feature: torch.Tensor) -> torch.Tensor:
        if rgb_feature.shape[-1] != self.config.rgb_feature_dim:
            raise ValueError(
                f"Expected rgb_feature last dim {self.config.rgb_feature_dim}, "
                f"got {rgb_feature.shape[-1]}"
            )
        if depth_feature.shape[-1] != self.config.depth_feature_dim:
            raise ValueError(
                f"Expected depth_feature last dim {self.config.depth_feature_dim}, "
                f"got {depth_feature.shape[-1]}"
            )

        fused = torch.cat([rgb_feature, depth_feature], dim=-1)
        return self.action_head(fused)
