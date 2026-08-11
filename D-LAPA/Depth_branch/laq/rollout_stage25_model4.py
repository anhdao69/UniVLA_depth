#!/usr/bin/env python3
"""
Stage 2.5 online rollout feature extractor.

Combines, in a single process, the two batch scripts that are normally
chained through disk (JSONL / .pt feature shards):

    latent_pretraining/inference_update_jsonl_train.py
        rgb_image + instruction --[LAPA]--> z_rgb_feature

    laq/test_ssv2_25_model4_no_gt.py
        depth_image + z_rgb_feature --[Model4]--> z_depth_feature_pred

Use this when you need one depth feature per step inside a live rollout
loop (e.g. LIBERO), instead of pre-extracting a dataset to disk.

Run from the laq/ directory so that both `laq_model` (sibling folder) and
`latent_pretraining` (sibling of laq/, one level up) resolve as packages.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
from typing import Optional, Union

import cv2
import numpy as np
import torch

from tux import JaxDistributedConfig, set_random_seed
from latent_pretraining.delta_llama import VideoLLaMAConfig
from latent_pretraining.inference_update_jsonl_train import LAPAInference

from laq_model.latent_action_quantization_stage25_feature_model4 import (
    LatentActionQuantizationStage25Model4,
)
from laq_model.latent_action_quantization_stage25_feature_model2 import (
    LatentActionQuantizationStage25Model2,
)
from laq_model.latent_action_quantization_stage25_feature_model5 import (
    LatentActionQuantizationStage25Model5,
)
from test_ssv2_25_model4_no_gt import (
    load_depth_image,
    load_model4_checkpoint,
    infer_output_config_from_checkpoint,
)


def build_lapa(
    tokens_per_delta: int,
    vqgan_checkpoint: str,
    vocab_file: str,
    load_checkpoint: str,
    image_size: int = 256,
    multi_image: int = 1,
    seed: int = 1234,
    mesh_dim: str = "1,1,1,1",
    dtype: str = "bf16",
    load_llama_config: str = "7b",
    update_llama_config: str = (
        "dict(delta_vocab_size=8,"
        "sample_mode='text',"
        "theta=50000000,"
        "max_sequence_length=32768,"
        "scan_attention=False,"
        "scan_query_chunk_size=128,"
        "scan_key_chunk_size=128,"
        "scan_mlp=False,"
        "scan_mlp_chunk_size=8192,"
        "scan_layers=True)"
    ),
    jax_distributed: Optional[dict] = None,
) -> LAPAInference:
    """Build the RGB -> z_rgb_feature model, mirroring inference_update_jsonl_train.main()."""

    jax_distributed = jax_distributed or JaxDistributedConfig.get_default_config()

    tokenizer_config = VideoLLaMAConfig.get_tokenizer_config()
    tokenizer_config.vocab_file = vocab_file
    llama_config = VideoLLaMAConfig.get_default_config()

    JaxDistributedConfig.initialize(jax_distributed)
    set_random_seed(seed)

    return LAPAInference(
        image_size=image_size,
        tokens_per_delta=tokens_per_delta,
        vqgan_checkpoint=vqgan_checkpoint,
        vocab_file=vocab_file,
        multi_image=multi_image,
        jax_distributed=jax_distributed,
        seed=seed,
        mesh_dim=mesh_dim,
        dtype=dtype,
        load_llama_config=load_llama_config,
        update_llama_config=update_llama_config,
        load_checkpoint=load_checkpoint,
        tokenizer=tokenizer_config,
        llama=llama_config,
    )


def build_model4(
    checkpoint: str,
    dim: int = 1024,
    image_size: int = 256,
    patch_size: int = 32,
    spatial_depth: int = 8,
    dim_head: int = 64,
    heads: int = 16,
    code_seq_len: int = 4,
    z_rgb_feature_dim: int = 4096,
    z_depth_feature_dim: int = 1024,
    predict_token_features: bool = False,
    strict: bool = True,
) -> LatentActionQuantizationStage25Model4:
    """Build + load the depth+RGB-feature -> z_depth_feature model, mirroring test_ssv2_25_model4_no_gt.main()."""

    ckpt_raw = torch.load(checkpoint, map_location="cpu")
    z_depth_feature_dim, predict_token_features = infer_output_config_from_checkpoint(
        ckpt_raw,
        default_dim=z_depth_feature_dim,
        default_predict_token_features=predict_token_features,
    )

    model = LatentActionQuantizationStage25Model4(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        spatial_depth=spatial_depth,
        dim_head=dim_head,
        heads=heads,
        code_seq_len=code_seq_len,
        z_rgb_feature_dim=z_rgb_feature_dim,
        z_depth_feature_dim=z_depth_feature_dim,
        predict_token_features=predict_token_features,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
    ).cuda()

    load_result, _ = load_model4_checkpoint(model, checkpoint, strict=strict)
    print("model4 checkpoint load result:", load_result)
    model.eval()

    return model


def build_model2(
    checkpoint: str,
    dim: int = 1024,
    image_size: int = 256,
    patch_size: int = 32,
    spatial_depth: int = 8,
    dim_head: int = 64,
    heads: int = 16,
    code_seq_len: int = 4,
    z_rgb_feature_dim: int = 4096,
    z_depth_feature_dim: int = 1024,
    predict_token_features: bool = False,
    strict: bool = True,
) -> LatentActionQuantizationStage25Model2:
    """Build Model2 and expose its fused 1024-D representation for Stage 3."""

    ckpt_raw = torch.load(checkpoint, map_location="cpu")
    state = ckpt_raw["model"] if isinstance(ckpt_raw, dict) and "model" in ckpt_raw else ckpt_raw
    head_weight = state.get("head.weight")
    codebook_size = int(head_weight.shape[0]) if head_weight is not None else 8

    model = LatentActionQuantizationStage25Model2(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        spatial_depth=spatial_depth,
        dim_head=dim_head,
        heads=heads,
        code_seq_len=code_seq_len,
        z_rgb_feature_dim=z_rgb_feature_dim,
        z_depth_feature_dim=z_depth_feature_dim,
        predict_token_features=predict_token_features,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
        codebook_size=codebook_size,
    ).cuda()

    load_result = model.load(checkpoint, strict=strict)
    print("model2 checkpoint load result:", load_result)
    model.eval()
    return model


def build_model5(
    checkpoint: str,
    dim: int = 1024,
    code_seq_len: int = 4,
    z_rgb_feature_dim: int = 4096,
    z_depth_feature_dim: int = 1024,
    predict_token_features: bool = False,
    strict: bool = True,
    **unused_kwargs,
) -> LatentActionQuantizationStage25Model5:
    """Build the RGB-feature-only Model5 with strict checkpoint validation."""
    ckpt_raw = torch.load(checkpoint, map_location="cpu")
    z_depth_feature_dim, predict_token_features = infer_output_config_from_checkpoint(
        ckpt_raw,
        default_dim=z_depth_feature_dim,
        default_predict_token_features=predict_token_features,
    )
    model = LatentActionQuantizationStage25Model5(
        dim=dim,
        code_seq_len=code_seq_len,
        z_rgb_feature_dim=z_rgb_feature_dim,
        z_depth_feature_dim=z_depth_feature_dim,
        predict_token_features=predict_token_features,
    ).cuda()
    load_result = model.load(checkpoint, strict=strict)
    print("model5 checkpoint load result:", load_result)
    model.eval()
    return model


class Stage25RolloutFeatureExtractor:
    """
    Online (per-step) replacement for the two batch scripts:

        rgb_image + instruction --[LAPA]--> z_rgb_feature
        depth_image + z_rgb_feature --[Model4]--> z_depth_feature_pred

    Nothing is written to disk; call `step()` once per rollout timestep.
    """

    def __init__(
        self,
        lapa: LAPAInference,
        model4: LatentActionQuantizationStage25Model4,
        model_name: str = "model4",
        image_size: int = 256,
        depth_scale: float = 65535.0,
        repeat_depth_to_3ch: bool = True,
    ):
        self.lapa = lapa
        self.model4 = model4
        self.model_name = model_name
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.repeat_depth_to_3ch = repeat_depth_to_3ch

    def _depth_to_tensor(self, depth_image: Union[str, "os.PathLike[str]", np.ndarray]) -> torch.Tensor:
        if isinstance(depth_image, (str, os.PathLike)):
            return load_depth_image(
                path=str(depth_image),
                image_size=self.image_size,
                repeat_depth_to_3ch=self.repeat_depth_to_3ch,
                depth_scale=self.depth_scale,
            )

        depth = np.asarray(depth_image)
        if depth.ndim == 3:
            depth = depth[..., 0] if depth.shape[-1] != 3 else cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = depth.astype("float32") / float(self.depth_scale)
        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(depth).float().unsqueeze(0)  # [1,H,W]
        if self.repeat_depth_to_3ch:
            x = x.repeat(3, 1, 1)  # [3,H,W]
        return x

    @torch.no_grad()
    def step(
        self,
        rgb_image: np.ndarray,
        instruction: str,
        depth_image: Union[str, "os.PathLike[str]", np.ndarray],
    ) -> dict:
        """
        Args:
            rgb_image: HxWx3 uint8 array, current RGB observation.
            instruction: task instruction text (fixed per episode).
            depth_image: path to a depth image file, or an already-loaded
                HxW / HxWx1 depth array (uint16 or float), current depth observation.

        Returns:
            {
                "latent_action": ...,              # raw LAPA output, unchanged
                "z_rgb_feature": FloatTensor[D_rgb],
                "z_depth_feature_pred": FloatTensor[D_depth] (or [L, D_depth]
                    if the checkpoint was trained with predict_token_features),
            }
        """
        latent_action, z_rgb_feature = self.lapa.inference(
            rgb_image,
            instruction,
            return_feature=True,
            feature_pool="mean",
        )

        z_rgb_features = torch.as_tensor(
            np.asarray(z_rgb_feature, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0).cuda()

        if self.model_name == "model5":
            z_depth_feature_pred = self.model4.extract_z_depth_feature(
                z_rgb_features=z_rgb_features,
            )
        else:
            depth1 = self._depth_to_tensor(depth_image).unsqueeze(0).cuda().float()
            z_depth_feature_pred = self.model4.extract_z_depth_feature(
                depth1=depth1,
                z_rgb_features=z_rgb_features,
            )

        return {
            "latent_action": latent_action,
            "z_rgb_feature": z_rgb_features.squeeze(0).detach().cpu(),
            "z_depth_feature_pred": z_depth_feature_pred.squeeze(0).detach().cpu(),
        }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test: one rollout step through LAPA "
            "(rgb+instruction->z_rgb_feature) + Model4 "
            "(depth+z_rgb_feature->z_depth_feature_pred)."
        )
    )

    # single-sample demo inputs
    parser.add_argument("--rgb_image", type=str, required=True)
    parser.add_argument("--depth_image", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--output_pt", type=str, default="")

    # LAPA
    parser.add_argument("--tokens_per_delta", type=int, default=4)
    parser.add_argument("--vqgan_checkpoint", type=str, default="lapa_checkpoints/vqgan")
    parser.add_argument("--vocab_file", type=str, default="lapa_checkpoints/tokenizer.model")
    parser.add_argument("--load_checkpoint", type=str, default="params::lapa_checkpoints/streaming_params_22485")
    parser.add_argument("--multi_image", type=int, default=1)
    parser.add_argument("--mesh_dim", type=str, default="1,1,1,1")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--load_llama_config", type=str, default="7b")
    parser.add_argument("--seed", type=int, default=1234)

    # Model4
    parser.add_argument("--model4_checkpoint", type=str, required=True)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--spatial_depth", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--code_seq_len", type=int, default=4)
    parser.add_argument("--z_rgb_feature_dim", type=int, default=4096)
    parser.add_argument("--z_depth_feature_dim", type=int, default=1024)
    parser.add_argument("--predict_token_features", type=int, default=0, choices=[0, 1])
    parser.add_argument("--strict", type=int, default=1, choices=[0, 1])

    # shared
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--repeat_depth_to_3ch", type=int, default=1, choices=[0, 1])
    parser.add_argument("--depth_scale", type=float, default=65535.0)

    args = parser.parse_args()

    lapa = build_lapa(
        tokens_per_delta=args.tokens_per_delta,
        vqgan_checkpoint=args.vqgan_checkpoint,
        vocab_file=args.vocab_file,
        load_checkpoint=args.load_checkpoint,
        image_size=args.image_size,
        multi_image=args.multi_image,
        seed=args.seed,
        mesh_dim=args.mesh_dim,
        dtype=args.dtype,
        load_llama_config=args.load_llama_config,
    )

    model4 = build_model4(
        checkpoint=args.model4_checkpoint,
        dim=args.dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        spatial_depth=args.spatial_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        code_seq_len=args.code_seq_len,
        z_rgb_feature_dim=args.z_rgb_feature_dim,
        z_depth_feature_dim=args.z_depth_feature_dim,
        predict_token_features=bool(args.predict_token_features),
        strict=bool(args.strict),
    )

    extractor = Stage25RolloutFeatureExtractor(
        lapa=lapa,
        model4=model4,
        image_size=args.image_size,
        depth_scale=args.depth_scale,
        repeat_depth_to_3ch=bool(args.repeat_depth_to_3ch),
    )

    from PIL import Image

    rgb = np.array(
        Image.open(args.rgb_image).convert("RGB").resize((args.image_size, args.image_size)),
        dtype=np.uint8,
    )

    out = extractor.step(rgb, args.instruction, args.depth_image)

    print("z_rgb_feature:", tuple(out["z_rgb_feature"].shape))
    print("z_depth_feature_pred:", tuple(out["z_depth_feature_pred"].shape))

    if args.output_pt:
        torch.save(out, args.output_pt)
        print("Saved:", args.output_pt)


if __name__ == "__main__":
    main()
