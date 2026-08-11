from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig


@dataclass
class VQResult:
    quantized: Tensor
    embeddings: Tensor
    prequant: Tensor
    indices: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor


class VectorQuantizer(nn.Module):
    """FP32 nearest-neighbor VQ with a straight-through encoder gradient."""

    def __init__(self, num_codes: int, code_dim: int) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.codebook = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.num_codes, 1.0 / self.num_codes)
        self.register_buffer("usage", torch.zeros(self.num_codes, dtype=torch.long), persistent=False)

    def forward(self, x: Tensor) -> VQResult:
        original_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            weight32 = self.codebook.weight.float()
            distances = (
                x32.square().sum(dim=-1, keepdim=True)
                - 2.0 * x32 @ weight32.t()
                + weight32.square().sum(dim=-1)
            )
            indices = distances.argmin(dim=-1)
            embeddings = F.embedding(indices, weight32)
            codebook_loss = F.mse_loss(embeddings, x32.detach())
            commitment_loss = F.mse_loss(x32, embeddings.detach())
            quantized32 = x32 + (embeddings - x32).detach()

        if self.training:
            with torch.no_grad():
                self.usage.add_(torch.bincount(indices.reshape(-1), minlength=self.num_codes))

        return VQResult(
            quantized=quantized32.to(original_dtype),
            embeddings=embeddings,
            prequant=x32,
            indices=indices,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )

    @torch.no_grad()
    def synchronized_usage(self) -> Tensor:
        usage = self.usage.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(usage, op=dist.ReduceOp.SUM)
        return usage

    @torch.no_grad()
    def reset_usage(self) -> None:
        self.usage.zero_()

    @torch.no_grad()
    def restart_dead(self, optimizer: torch.optim.Optimizer, noise_scale: float = 1e-4) -> int:
        usage = self.synchronized_usage()
        dead = torch.where(usage == 0)[0]
        active = torch.where(usage > 0)[0]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0 and dead.numel():
            if active.numel():
                source = active[torch.randint(active.numel(), (dead.numel(),), device=active.device)]
                replacement = self.codebook.weight[source].clone()
                replacement.add_(torch.randn_like(replacement) * noise_scale)
                self.codebook.weight[dead] = replacement
            else:
                nn.init.uniform_(self.codebook.weight, -1.0 / self.num_codes, 1.0 / self.num_codes)

        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.codebook.weight, src=0)
        if dead.numel():
            state = optimizer.state.get(self.codebook.weight, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(name)
                if torch.is_tensor(value) and value.shape == self.codebook.weight.shape:
                    if active.numel():
                        value[dead] = 0
                    else:
                        value.zero_()
        self.reset_usage()
        return int(dead.numel())


class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        b, t, s, d = x.shape
        spatial = self.spatial_norm(x).reshape(b * t, s, d)
        spatial_mask = padding_mask.reshape(b * t, s) if padding_mask is not None else None
        spatial, _ = self.spatial_attn(
            spatial, spatial, spatial, key_padding_mask=spatial_mask, need_weights=False
        )
        x = x + spatial.reshape(b, t, s, d)

        temporal = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(b * s, t, d)
        causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        temporal, _ = self.temporal_attn(
            temporal, temporal, temporal, attn_mask=causal_mask, need_weights=False
        )
        x = x + temporal.reshape(b, s, t, d).permute(0, 2, 1, 3)
        x = x + self.ff(self.ff_norm(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask[..., None], 0)
        return x


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.encoder_depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        for block in self.blocks:
            x = block(x, padding_mask)
        return self.norm(x)


class SpatialDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.dim * cfg.mlp_ratio,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=cfg.decoder_depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        return self.norm(self.layers(x, src_key_padding_mask=padding_mask))


class DepthFactorizedLAM(nn.Module):
    """Two-frame depth LAM implementing the faithful UniVLA stage semantics."""

    TRANSFER_PREFIXES = (
        "patch_norm_in.", "patch_projection.", "patch_norm_out.", "spatial_pos",
        "temporal_pos", "encoder.", "decoder.", "ti_queries", "ti_to_code.",
        "ti_codebook.", "ti_from_code.", "patch_head.",
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        patch_dim = config.patch_size * config.patch_size
        self.patch_norm_in = nn.LayerNorm(patch_dim)
        self.patch_projection = nn.Linear(patch_dim, config.dim)
        self.patch_norm_out = nn.LayerNorm(config.dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, config.num_patches, config.dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, 2, 1, config.dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        self.encoder = SpatioTemporalEncoder(config)
        self.decoder = SpatialDecoder(config)
        self.ti_queries = nn.Parameter(torch.empty(1, 1, config.num_queries, config.dim))
        nn.init.uniform_(self.ti_queries, -1.0, 1.0)
        self.ti_to_code = nn.Linear(config.dim, config.vq_dim)
        self.ti_codebook = VectorQuantizer(config.irr_codebook_size, config.vq_dim)
        self.ti_from_code = nn.Linear(config.vq_dim, config.dim)

        if config.stage == "irr":
            self.language_proj: Optional[nn.Module] = nn.Linear(config.text_dim, config.dim)
            self.tc_queries = None
            self.tc_to_code = None
            self.tc_codebook = None
            self.tc_from_code = None
        else:
            self.language_proj = None
            self.tc_queries = nn.Parameter(torch.empty(1, 1, config.num_queries, config.dim))
            nn.init.uniform_(self.tc_queries, -1.0, 1.0)
            self.tc_to_code = nn.Linear(config.dim, config.vq_dim)
            self.tc_codebook = VectorQuantizer(config.task_codebook_size, config.vq_dim)
            self.tc_from_code = nn.Linear(config.vq_dim, config.dim)

        self.patch_head = nn.Linear(config.dim, patch_dim)

    def _patchify(self, depth: Tensor) -> Tensor:
        b, t, c, h, w = depth.shape
        if (t, c, h, w) != (2, 1, self.config.image_size, self.config.image_size):
            raise ValueError(
                f"expected [B,2,1,{self.config.image_size},{self.config.image_size}], got {tuple(depth.shape)}"
            )
        flat = depth.reshape(b * t, c, h, w)
        patches = F.unfold(flat, kernel_size=self.config.patch_size, stride=self.config.patch_size)
        patches = patches.transpose(1, 2)
        patches = self.patch_norm_out(self.patch_projection(self.patch_norm_in(patches)))
        return patches.reshape(b, t, self.config.num_patches, self.config.dim)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        b = patches.shape[0]
        pixels = patches.transpose(1, 2)
        return F.fold(
            pixels,
            output_size=(self.config.image_size, self.config.image_size),
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        ).reshape(b, 1, self.config.image_size, self.config.image_size)

    def _language_tokens(self, language: Tensor) -> Tensor:
        if self.language_proj is None:
            raise RuntimeError("language is unavailable in the task-centric stage")
        return self.language_proj(language)

    def forward(
        self,
        depth_pair: Tensor,
        language: Optional[Tensor] = None,
        language_mask: Optional[Tensor] = None,
        task_ablation: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        if self.config.stage == "task" and (language is not None or language_mask is not None):
            raise ValueError("Stage 1b must not receive language")
        if self.config.stage == "irr" and (language is None or language_mask is None):
            raise ValueError("Stage 1a requires token-level language and a mask")

        patches = self._patchify(depth_pair)
        patches = patches + self.spatial_pos + self.temporal_pos
        b = patches.shape[0]
        ti = self.ti_queries.expand(b, 2, -1, -1)

        if self.config.stage == "irr":
            visual = torch.cat([ti, patches], dim=2)
            lang = self._language_tokens(language)
            lang = lang[:, None].expand(-1, 2, -1, -1)
            encoded_input = torch.cat([visual, lang], dim=2)
            visual_padding = torch.zeros(
                b, 2, visual.shape[2], dtype=torch.bool, device=depth_pair.device
            )
            lang_padding = ~language_mask.bool()[:, None].expand(-1, 2, -1)
            padding_mask = torch.cat([visual_padding, lang_padding], dim=2)
            encoded = self.encoder(encoded_input, padding_mask)
            ti_tokens = encoded[:, 1, : self.config.num_queries]
            tc_tokens = None
        else:
            assert self.tc_queries is not None
            tc = self.tc_queries.expand(b, 2, -1, -1)
            encoded_input = torch.cat([tc, ti, patches], dim=2)
            encoded = self.encoder(encoded_input, None)
            tc_tokens = encoded[:, 1, : self.config.num_queries]
            ti_tokens = encoded[:, 1, self.config.num_queries : 2 * self.config.num_queries]

        ti_vq = self.ti_codebook(self.ti_to_code(ti_tokens))
        q_ti = self.ti_from_code(ti_vq.quantized)
        output: Dict[str, Tensor] = {
            "irr_tokens": ti_tokens,
            "irr_feature": ti_tokens.float().mean(dim=1),
            "irr_indices": ti_vq.indices,
            "irr_quantized": ti_vq.quantized,
            "irr_codebook_loss": ti_vq.codebook_loss,
            "irr_commitment_loss": ti_vq.commitment_loss,
        }

        current_patches = patches[:, 0]
        if self.config.stage == "irr":
            decoder_visual = torch.cat([q_ti, current_patches], dim=1)
            decoder_lang = self._language_tokens(language)
            decoder_input = torch.cat([decoder_visual, decoder_lang], dim=1)
            decoder_padding = torch.cat(
                [
                    torch.zeros(b, decoder_visual.shape[1], dtype=torch.bool, device=depth_pair.device),
                    ~language_mask.bool(),
                ],
                dim=1,
            )
            decoded = self.decoder(decoder_input, decoder_padding)
            patch_start = self.config.num_queries
        else:
            assert tc_tokens is not None and self.tc_to_code is not None
            assert self.tc_codebook is not None and self.tc_from_code is not None
            tc_vq = self.tc_codebook(self.tc_to_code(tc_tokens))
            q_tc = self.tc_from_code(tc_vq.quantized)
            if task_ablation == "zero":
                q_tc = torch.zeros_like(q_tc)
            elif task_ablation == "shuffle":
                q_tc = q_tc.roll(1, dims=0)
            elif task_ablation is not None:
                raise ValueError(f"unknown task ablation {task_ablation!r}")
            decoder_input = torch.cat([q_tc, q_ti, current_patches], dim=1)
            decoded = self.decoder(decoder_input, None)
            patch_start = 2 * self.config.num_queries
            output.update(
                {
                    "task_tokens": tc_tokens,
                    "task_feature": tc_tokens.float().mean(dim=1),
                    "task_indices": tc_vq.indices,
                    "task_quantized": tc_vq.quantized,
                    "task_codebook_loss": tc_vq.codebook_loss,
                    "task_commitment_loss": tc_vq.commitment_loss,
                }
            )

        predicted_patches = self.patch_head(
            decoded[:, patch_start : patch_start + self.config.num_patches]
        )
        reconstruction = self._unpatchify(predicted_patches)
        reconstruction_loss = F.mse_loss(reconstruction.float(), depth_pair[:, 1].float())
        beta = self.config.commitment_beta
        if self.config.stage == "irr":
            loss = reconstruction_loss + ti_vq.codebook_loss + beta * ti_vq.commitment_loss
        else:
            loss = (
                reconstruction_loss
                + output["task_codebook_loss"]
                + beta * output["task_commitment_loss"]
                + beta * ti_vq.commitment_loss
            )
        output.update(
            {
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "reconstruction": reconstruction,
                "target": depth_pair[:, 1],
            }
        )
        return output

    def initialize_task_from_stage1(self, stage1_state: Dict[str, Tensor]) -> Dict[str, list[str]]:
        if self.config.stage != "task":
            raise RuntimeError("transfer is only valid for a task-stage model")
        own = self.state_dict()
        transferred: list[str] = []
        unexpected: list[str] = []
        for key, value in stage1_state.items():
            if key.startswith("language_proj."):
                continue
            if any(key == prefix or key.startswith(prefix) for prefix in self.TRANSFER_PREFIXES):
                if key not in own or own[key].shape != value.shape:
                    unexpected.append(key)
                    continue
                own[key].copy_(value)
                transferred.append(key)
        required = [
            key for key in own
            if any(key == prefix or key.startswith(prefix) for prefix in self.TRANSFER_PREFIXES)
            and not key.startswith("tc_")
        ]
        missing = sorted(set(required) - set(transferred))
        if missing or unexpected:
            raise RuntimeError(f"invalid Stage-1a transfer: missing={missing}, incompatible={unexpected}")
        self.ti_codebook.codebook.weight.requires_grad_(False)
        return {"transferred": sorted(transferred), "missing": missing, "incompatible": unexpected}

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def extract_teacher(self, depth_t: Tensor, depth_future: Tensor) -> Dict[str, Tensor]:
        if self.config.stage != "task":
            raise RuntimeError("teacher extraction requires the task-centric stage")
        was_training = self.training
        self.eval()
        outputs = self(torch.stack([depth_t, depth_future], dim=1))
        if was_training:
            self.train()
        keys = (
            "task_feature", "task_tokens", "task_indices", "task_quantized",
            "irr_feature", "irr_tokens", "irr_indices", "irr_quantized",
        )
        return {key: outputs[key] for key in keys}


@torch.no_grad()
def extract_legacy_pre_vq_feature(model: nn.Module, video: Tensor) -> Tensor:
    """Formalized 1024-D legacy target: mean spatial encoder difference."""
    if video.ndim != 5 or video.shape[2] != 2:
        raise ValueError("legacy video must have shape [B,C,2,H,W]")
    first = model.to_patch_emb_first_frame(video[:, :, :1])
    last = model.to_patch_emb_first_frame(video[:, :, 1:])
    first_tokens, last_tokens = model.encode(torch.cat([first, last], dim=1))
    return (last_tokens - first_tokens).mean(dim=(1, 2, 3)).float()
