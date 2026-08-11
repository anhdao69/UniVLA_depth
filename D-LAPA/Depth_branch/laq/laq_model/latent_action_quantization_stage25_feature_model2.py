from pathlib import Path

import torch
from torch import nn

from laq_model.latent_action_quantization_stage25_feature_model4 import (
    LatentActionQuantizationStage25Model4,
)


class LatentActionQuantizationStage25Model2(LatentActionQuantizationStage25Model4):
    """Model2 Stage-2.5 depth+RGB feature encoder.

    The released Model2 checkpoint uses the same depth/RGB/fusion trunk as
    Model4, but its checkpoint ends with a small classifier head
    (``head.weight`` / ``head.bias``) instead of Model4's continuous feature
    heads. For Stage-3 depth injection we need the compact 1024-D fused
    representation, so ``extract_z_depth_feature`` returns the fused feature
    before this classifier head.
    """

    def __init__(
        self,
        *,
        codebook_size=8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if hasattr(self, "slot_embed"):
            del self.slot_embed
        if hasattr(self, "slot_mlp"):
            del self.slot_mlp
        if hasattr(self, "feature_head_global"):
            del self.feature_head_global
        if hasattr(self, "feature_head_tokens"):
            del self.feature_head_tokens

        self.head = nn.Linear(self.dim, int(codebook_size))

    def load(self, path, strict=True):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location="cpu")
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
        return self.load_state_dict(state, strict=strict)

    def predict(self, depth1, z_rgb_features):
        fused_feature, _, _ = self.encode_fused_feature(
            depth1=depth1,
            z_rgb_features=z_rgb_features,
        )
        return fused_feature

    def forward(self, depth1, z_rgb_features, z_depth_feature=None):
        fused_feature = self.predict(depth1=depth1, z_rgb_features=z_rgb_features)
        logits = self.head(fused_feature)
        if z_depth_feature is None:
            return logits
        return logits

    def extract_z_depth_feature(self, depth1, z_rgb_features):
        return self.predict(depth1=depth1, z_rgb_features=z_rgb_features)
