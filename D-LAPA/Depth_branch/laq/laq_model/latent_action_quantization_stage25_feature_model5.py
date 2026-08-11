from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LatentActionQuantizationStage25Model5(nn.Module):
    """Model5: pooled RGB feature -> continuous depth feature.

    Unlike Model2/Model4, this model does not consume a depth image. The module
    names and dimensions intentionally match the released Model5 checkpoint.
    """

    def __init__(
        self,
        *,
        dim=1024,
        code_seq_len=4,
        z_rgb_feature_dim=4096,
        z_rgb_feature_dropout=0.0,
        z_depth_feature_dim=1024,
        predict_token_features=False,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
        **unused_kwargs,
    ):
        super().__init__()
        self.dim = int(dim)
        self.code_seq_len = int(code_seq_len)
        self.z_rgb_feature_dim = int(z_rgb_feature_dim)
        self.z_depth_feature_dim = int(z_depth_feature_dim)
        self.predict_token_features = bool(predict_token_features)
        self.feature_loss_weight = float(feature_loss_weight)
        self.cosine_loss_weight = float(cosine_loss_weight)

        self.slot_embed = nn.Parameter(
            torch.randn(self.code_seq_len, self.dim) * 0.02
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.z_rgb_feature_dim),
            nn.Linear(self.z_rgb_feature_dim, 2048),
            nn.GELU(),
            nn.Dropout(z_rgb_feature_dropout),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(z_rgb_feature_dropout),
            nn.Linear(2048, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.feature_head_global = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.z_depth_feature_dim),
        )
        self.feature_head_tokens = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.z_depth_feature_dim),
        )

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

    def predict(self, z_rgb_features):
        if z_rgb_features.ndim != 2:
            raise ValueError(
                f"Expected RGB features [B,D], got {tuple(z_rgb_features.shape)}"
            )
        if z_rgb_features.shape[1] != self.z_rgb_feature_dim:
            raise ValueError(
                f"Expected RGB feature dim {self.z_rgb_feature_dim}, "
                f"got {z_rgb_features.shape[1]}"
            )
        encoded = self.encoder(z_rgb_features.float())
        if self.predict_token_features:
            slots = encoded[:, None, :] + self.slot_embed[None, :, :]
            return self.feature_head_tokens(self.slot_mlp(slots))
        return self.feature_head_global(encoded)

    def compute_feature_loss(self, prediction, target):
        target = target.float()
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Prediction/target mismatch: {tuple(prediction.shape)} "
                f"versus {tuple(target.shape)}"
            )
        mse = F.mse_loss(prediction, target)
        cosine = 1.0 - F.cosine_similarity(
            prediction.reshape(prediction.shape[0], -1),
            target.reshape(target.shape[0], -1),
            dim=-1,
        ).mean()
        return mse, cosine

    def forward(self, z_rgb_features, z_depth_feature=None):
        prediction = self.predict(z_rgb_features)
        if z_depth_feature is None:
            return prediction
        mse, cosine = self.compute_feature_loss(prediction, z_depth_feature)
        loss = self.feature_loss_weight * mse + self.cosine_loss_weight * cosine
        return loss, {
            "loss": loss.detach(),
            "feature_mse_loss": mse.detach(),
            "feature_cosine_loss": cosine.detach(),
        }, prediction

    def extract_z_depth_feature(self, z_rgb_features):
        return self.predict(z_rgb_features)
