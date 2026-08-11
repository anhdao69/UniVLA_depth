from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

from laq_model.attention import Transformer, ContinuousPositionBias


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class LatentActionQuantizationStage25Model4(nn.Module):
    """
    Model 4 for LAPA-depth Stage 2.5.

    Pipeline:
        depth1 + z_rgb_features -> z_depth_feature

    This version:
        - does NOT use z_depth_indices
        - does NOT predict codebook IDs
        - trains only with continuous z_depth_feature ground truth

    Inputs:
        depth1:         FloatTensor [B, C, H, W]
        z_rgb_features: FloatTensor [B, z_rgb_feature_dim], usually [B, 4096]

    Target:
        z_depth_feature:
            FloatTensor [B, z_depth_feature_dim]
            or FloatTensor [B, code_seq_len, z_depth_feature_dim]
            if predict_token_features=True

    Outputs:
        If z_depth_feature is given:
            loss, logs, pred_z_depth_feature
        Else:
            pred_z_depth_feature

    Typical configs:
        If Stage 1 returns pooled VQ feature:
            z_depth_feature_dim=32
            predict_token_features=False

        If Stage 1 returns global encoder feature:
            z_depth_feature_dim=1024
            predict_token_features=False

        If Stage 1 returns token-level feature:
            z_depth_feature_dim=32 or 1024
            predict_token_features=True
    """

    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        spatial_depth,
        dim_head=64,
        heads=8,
        channels=3,
        attn_dropout=0.0,
        ff_dropout=0.0,
        code_seq_len=4,
        z_rgb_feature_dim=4096,
        z_rgb_feature_dropout=0.0,
        z_depth_feature_dim=1024,
        predict_token_features=False,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
        **unused_kwargs,
    ):
        """
        unused_kwargs lets old scripts pass unused args such as codebook_size
        without breaking Model 4.
        """
        super().__init__()

        self.dim = dim
        self.code_seq_len = int(code_seq_len)
        self.z_rgb_feature_dim = int(z_rgb_feature_dim)
        self.z_depth_feature_dim = int(z_depth_feature_dim)
        self.predict_token_features = bool(predict_token_features)

        self.feature_loss_weight = float(feature_loss_weight)
        self.cosine_loss_weight = float(cosine_loss_weight)

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)

        patch_height, patch_width = self.patch_size
        image_height, image_width = self.image_size

        assert image_height % patch_height == 0
        assert image_width % patch_width == 0

        self.spatial_rel_pos_bias = ContinuousPositionBias(
            dim=dim,
            heads=heads,
        )

        self.to_patch_emb_depth = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b h w (c p1 p2)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(channels * patch_height * patch_width),
            nn.Linear(channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )

        self.depth_spatial_transformer = Transformer(
            depth=spatial_depth,
            **transformer_kwargs,
        )

        self.z_rgb_feature_proj = nn.Sequential(
            nn.LayerNorm(z_rgb_feature_dim),
            nn.Linear(z_rgb_feature_dim, dim),
            nn.GELU(),
            nn.Dropout(z_rgb_feature_dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        # Used only if teacher z_depth_feature is token-level:
        # target shape [B, code_seq_len, z_depth_feature_dim].
        self.slot_embed = nn.Parameter(torch.randn(self.code_seq_len, dim) * 0.02)

        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.feature_head_global = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, z_depth_feature_dim),
        )

        self.feature_head_tokens = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, z_depth_feature_dim),
        )

    @property
    def patch_height_width(self):
        return (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

    def load(self, path, strict=False):
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"

        pt = torch.load(str(path), map_location="cpu")

        if isinstance(pt, dict) and "model" in pt:
            pt = pt["model"]

        pt = {
            k.replace("module.", "") if "module." in k else k: v
            for k, v in pt.items()
        }

        return self.load_state_dict(pt, strict=strict)

    def encode_depth1(self, depth1):
        """
        Encode current depth frame.

        Args:
            depth1: [B, C, H, W]

        Returns:
            depth_feature: [B, D]
            depth_tokens:  [B, H_patch, W_patch, D]
        """
        assert depth1.ndim == 4, f"Expected [B, C, H, W], got {depth1.shape}"

        b, c, h_img, w_img = depth1.shape
        assert (h_img, w_img) == self.image_size, (
            f"Expected image size {self.image_size}, got {(h_img, w_img)}"
        )

        h, w = self.patch_height_width

        depth_tokens = self.to_patch_emb_depth(depth1)  # [B, H, W, D]
        depth_tokens = rearrange(depth_tokens, "b h w d -> b 1 h w d")

        video_shape = tuple(depth_tokens.shape[:-1])  # (B, 1, H, W)
        tokens = rearrange(depth_tokens, "b t h w d -> (b t) (h w) d")

        attn_bias = self.spatial_rel_pos_bias(
            h,
            w,
            device=tokens.device,
        )

        tokens = self.depth_spatial_transformer(
            tokens,
            attn_bias=attn_bias,
            video_shape=video_shape,
        )

        depth_tokens = rearrange(
            tokens,
            "(b t) (h w) d -> b t h w d",
            b=b,
            h=h,
            w=w,
        )[:, 0]  # [B, H, W, D]

        depth_feature = depth_tokens.mean(dim=(1, 2))  # [B, D]

        return depth_feature, depth_tokens

    def encode_z_rgb_features(self, z_rgb_features):
        """
        Project precomputed LAPA/VLA RGB feature into model dim.

        Args:
            z_rgb_features: [B, z_rgb_feature_dim], e.g. [B, 4096]

        Returns:
            z_rgb_feature: [B, D]
        """
        assert z_rgb_features.ndim == 2, (
            f"Expected [B, z_rgb_feature_dim], got {z_rgb_features.shape}"
        )
        assert z_rgb_features.shape[1] == self.z_rgb_feature_dim, (
            f"Expected z_rgb_feature_dim={self.z_rgb_feature_dim}, "
            f"got {z_rgb_features.shape[1]}"
        )

        return self.z_rgb_feature_proj(z_rgb_features.float())

    def encode_fused_feature(self, depth1, z_rgb_features):
        depth_feature, depth_tokens = self.encode_depth1(depth1)
        z_rgb_feature = self.encode_z_rgb_features(z_rgb_features)

        fused = torch.cat([depth_feature, z_rgb_feature], dim=-1)
        fused_feature = self.fusion(fused)

        return fused_feature, depth_feature, z_rgb_feature

    def predict(self, depth1, z_rgb_features):
        """
        Predict continuous z_depth_feature.

        Returns:
            pred_z_depth_feature:
                [B, z_depth_feature_dim] if predict_token_features=False
                [B, code_seq_len, z_depth_feature_dim] if predict_token_features=True
        """
        fused_feature, depth_feature, z_rgb_feature = self.encode_fused_feature(
            depth1=depth1,
            z_rgb_features=z_rgb_features,
        )

        if self.predict_token_features:
            slot_tokens = fused_feature[:, None, :] + self.slot_embed[None, :, :]
            slot_tokens = self.slot_mlp(slot_tokens)
            pred_z_depth_feature = self.feature_head_tokens(slot_tokens)
        else:
            pred_z_depth_feature = self.feature_head_global(fused_feature)

        return pred_z_depth_feature

    def compute_feature_loss(self, pred_z_depth_feature, gt_z_depth_feature):
        gt_z_depth_feature = gt_z_depth_feature.float()

        if pred_z_depth_feature.shape != gt_z_depth_feature.shape:
            raise RuntimeError(
                "pred_z_depth_feature and gt_z_depth_feature shape mismatch: "
                f"pred={tuple(pred_z_depth_feature.shape)}, "
                f"gt={tuple(gt_z_depth_feature.shape)}. "
                "Set z_depth_feature_dim and predict_token_features correctly."
            )

        mse_loss = F.mse_loss(pred_z_depth_feature, gt_z_depth_feature)

        pred_flat = pred_z_depth_feature.reshape(pred_z_depth_feature.shape[0], -1)
        gt_flat = gt_z_depth_feature.reshape(gt_z_depth_feature.shape[0], -1)
        cosine_loss = 1.0 - F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean()

        return mse_loss, cosine_loss

    def forward(
        self,
        depth1,
        z_rgb_features,
        z_depth_feature=None,
    ):
        """
        Args:
            depth1:          [B, C, H, W]
            z_rgb_features:  [B, z_rgb_feature_dim]
            z_depth_feature: [B, D] or [B, L, D], optional

        Returns:
            If z_depth_feature is provided:
                loss, logs, pred_z_depth_feature
            Else:
                pred_z_depth_feature
        """
        pred_z_depth_feature = self.predict(
            depth1=depth1,
            z_rgb_features=z_rgb_features,
        )

        if z_depth_feature is None:
            return pred_z_depth_feature

        mse_loss, cosine_loss = self.compute_feature_loss(
            pred_z_depth_feature=pred_z_depth_feature,
            gt_z_depth_feature=z_depth_feature,
        )

        loss = self.feature_loss_weight * mse_loss
        loss = loss + self.cosine_loss_weight * cosine_loss

        logs = {
            "loss": loss.detach(),
            "feature_mse_loss": mse_loss.detach(),
            "feature_cosine_loss": cosine_loss.detach(),
        }

        return loss, logs, pred_z_depth_feature

    def extract_z_depth_feature(self, depth1, z_rgb_features):
        return self.predict(
            depth1=depth1,
            z_rgb_features=z_rgb_features,
        )
