#!/usr/bin/env python3
"""
Inference-only Model4 feature generation.

This version does NOT require z_depth_feature_manifest / ground truth depth features.
It loads:
  1) depth jsonl -> depth1 image tensor
  2) RGB feature manifest -> z_rgb_features tensor
Then generates:
  z_depth_feature_pred = model.extract_z_depth_feature(depth1, z_rgb_features)

Assumption: depth jsonl order and RGB feature manifest order are aligned.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from laq_model.latent_action_quantization_stage25_feature_model4 import (
    LatentActionQuantizationStage25Model4,
)


MODEL_NAME = "model4"
STAGE_NAME = "stage25_infer_no_gt"
DATASET_NAME = "libero_infer"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def find_depth_path(item):
    # In this depth JSONL, "image" is the depth image path.
    if "image" in item and item["image"]:
        return item["image"]

    for key in ["depth1_path", "depth_path", "depth", "image_path", "path"]:
        if key in item and item[key]:
            return item[key]

    raise KeyError(
        f"Cannot find depth path in jsonl item. "
        f"Available keys: {list(item.keys())}"
    )


def get_sample_id(item: Dict[str, Any], idx: int) -> str:
    for k in ["id", "sample_id", "uid", "video_id", "frame_id"]:
        if k in item:
            return str(item[k])
    return str(idx)


def load_depth_image(path: str, image_size: int, repeat_depth_to_3ch: bool, depth_scale: float) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Depth image not found: {p}")

    depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {p}")

    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = depth.astype("float32") / float(depth_scale)
    depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    x = torch.from_numpy(depth).float().unsqueeze(0)  # [1,H,W]
    if repeat_depth_to_3ch:
        x = x.repeat(3, 1, 1)  # [3,H,W]
    return x


def load_feature_manifest(manifest_path: str):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"RGB feature manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_dir = manifest_path.parent
    parts = manifest.get("parts", [])
    if not parts:
        raise RuntimeError(f"Manifest has no parts: {manifest_path}")

    preferred_key = manifest.get("feature_key", None)
    candidate_keys = []
    if preferred_key:
        candidate_keys.append(preferred_key)
    candidate_keys += [
        "z_rgb_features",
        "z_rgb_feature",
        "z_rgb_feature_pred",
        "z_rgb_feature_input",
        "features",
        "feature",
    ]

    all_features = []
    all_ids = []

    for part in parts:
        part_path = part.get("path")
        if part_path is None:
            raise KeyError(f"Part has no path: {part}")

        part_path = Path(part_path)
        if not part_path.is_absolute():
            part_path = base_dir / part_path

        pkg = torch.load(part_path, map_location="cpu")

        feature_key = None
        for k in candidate_keys:
            if k in pkg and torch.is_tensor(pkg[k]):
                feature_key = k
                break
        if feature_key is None:
            tensor_keys = [k for k, v in pkg.items() if torch.is_tensor(v)]
            raise KeyError(
                f"Cannot find RGB feature tensor in {part_path}. Tensor keys: {tensor_keys}"
            )

        feats = pkg[feature_key].detach().cpu().float()
        all_features.append(feats)

        ids = pkg.get("id", None)
        if ids is None:
            ids = [str(len(all_ids) + i) for i in range(feats.shape[0])]
        all_ids.extend([str(x) for x in ids])

    features = torch.cat(all_features, dim=0).float()
    return manifest, features, all_ids


class Stage25InferenceNoGTDataset(Dataset):
    def __init__(
        self,
        z_depth_path: str,
        z_rgb_feature_manifest: str,
        image_size: int = 256,
        repeat_depth_to_3ch: bool = True,
        depth_scale: float = 65535.0,
        check_length_alignment: bool = True,
    ):
        self.depth_items = load_jsonl(z_depth_path)
        self.rgb_manifest, self.z_rgb_features, self.rgb_ids = load_feature_manifest(
            z_rgb_feature_manifest
        )
        self.image_size = image_size
        self.repeat_depth_to_3ch = repeat_depth_to_3ch
        self.depth_scale = depth_scale

        if check_length_alignment and len(self.depth_items) != self.z_rgb_features.shape[0]:
            raise RuntimeError(
                "Length mismatch between depth jsonl and RGB features: "
                f"depth_jsonl={len(self.depth_items)}, "
                f"z_rgb_features={self.z_rgb_features.shape[0]}"
            )

        self.length = min(len(self.depth_items), self.z_rgb_features.shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        item = self.depth_items[idx]
        depth1_path = find_depth_path(item)
        depth1 = load_depth_image(
            path=depth1_path,
            image_size=self.image_size,
            repeat_depth_to_3ch=self.repeat_depth_to_3ch,
            depth_scale=self.depth_scale,
        )

        return {
            "id": get_sample_id(item, idx),
            "depth1_path": depth1_path,
            "depth1": depth1,
            "z_rgb_features": self.z_rgb_features[idx].float(),
        }


class FeaturePartWriter:
    def __init__(self, output_dir: str, prefix: str, part_size: int):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.part_size = part_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = {
            "id": [],
            "depth1_path": [],
            "z_depth_feature_pred": [],
            "z_rgb_feature_input": [],
        }
        self.parts = []
        self.part_idx = 0
        self.total_samples = 0

    def add(self, sample_id, depth1_path, z_depth_feature_pred, z_rgb_feature_input=None):
        self.buffer["id"].append(str(sample_id))
        self.buffer["depth1_path"].append(str(depth1_path))
        self.buffer["z_depth_feature_pred"].append(z_depth_feature_pred.detach().cpu().float())
        if z_rgb_feature_input is not None:
            self.buffer["z_rgb_feature_input"].append(z_rgb_feature_input.detach().cpu().float())
        self.flush(force=False)

    def flush(self, force=False):
        n = len(self.buffer["id"])
        if n == 0:
            return
        if not force and n < self.part_size:
            return

        z_depth_feature_pred = torch.stack(self.buffer["z_depth_feature_pred"], dim=0).float()
        out_path = self.output_dir / f"{self.prefix}_part{self.part_idx:05d}.pt"

        pkg = {
            "id": list(self.buffer["id"]),
            "depth1_path": list(self.buffer["depth1_path"]),
            "z_depth_feature_pred": z_depth_feature_pred,
            "model_name": MODEL_NAME,
            "stage": STAGE_NAME,
            "dataset": DATASET_NAME,
        }

        part_info = {
            "part": self.part_idx,
            "path": str(out_path),
            "num_samples": n,
            "z_depth_feature_pred_shape": list(z_depth_feature_pred.shape),
        }

        if len(self.buffer["z_rgb_feature_input"]) == n:
            z_rgb_feature_input = torch.stack(self.buffer["z_rgb_feature_input"], dim=0).float()
            pkg["z_rgb_feature_input"] = z_rgb_feature_input
            part_info["z_rgb_feature_input_shape"] = list(z_rgb_feature_input.shape)

        torch.save(pkg, out_path)
        self.parts.append(part_info)
        self.total_samples += n

        print(f"Saved feature part: {out_path} | samples={n} | pred={list(z_depth_feature_pred.shape)}")

        self.part_idx += 1
        for k in self.buffer:
            self.buffer[k].clear()

    def save_manifest(self, args):
        manifest = {
            "prefix": self.prefix,
            "total_samples": self.total_samples,
            "num_parts": len(self.parts),
            "feature_output_dir": str(self.output_dir),
            "checkpoint": args.checkpoint,
            "z_depth_path": args.z_depth_path,
            "z_rgb_feature_manifest": args.z_rgb_feature_manifest,
            "z_depth_feature_manifest": None,
            "uses_ground_truth": False,
            "model_name": MODEL_NAME,
            "stage": STAGE_NAME,
            "dataset": DATASET_NAME,
            "standardized_schema": True,
            "model_definition": "Model4 inference only: depth1 + z_rgb_features -> z_depth_feature_pred",
            "feature_key": "z_depth_feature_pred",
            "feature_key_pred": "z_depth_feature_pred",
            "rgb_feature_key_input": "z_rgb_feature_input",
            "parts": self.parts,
        }

        manifest_path = self.output_dir / f"{self.prefix}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Done. Wrote feature manifest to: {manifest_path}")


def load_model4_checkpoint(model, checkpoint_path: str, strict: bool = True):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {
        k.replace("module.", "") if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    return model.load_state_dict(state_dict, strict=strict), ckpt


def infer_output_config_from_checkpoint(ckpt: Any, default_dim: int, default_predict_token_features: bool):
    """Best-effort fallback. If training config is unavailable, use CLI defaults."""
    if isinstance(ckpt, dict):
        for cfg_key in ["args", "config", "model_args", "hparams"]:
            cfg = ckpt.get(cfg_key)
            if isinstance(cfg, dict):
                dim = cfg.get("z_depth_feature_dim", default_dim)
                ptf = cfg.get("predict_token_features", default_predict_token_features)
                return int(dim), bool(ptf)
    return int(default_dim), bool(default_predict_token_features)


def main():
    parser = argparse.ArgumentParser(
        description="Model4 inference only: depth1 + z_rgb_features -> z_depth_feature_pred, no GT needed."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--z_depth_path", type=str, required=True)
    parser.add_argument("--z_rgb_feature_manifest", type=str, required=True)

    parser.add_argument("--feature_output_dir", type=str, required=True)
    parser.add_argument("--feature_prefix", type=str, default="z_depth_model4")
    parser.add_argument("--feature_part_size", type=int, default=8192)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--save_rgb", action="store_true")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--repeat_depth_to_3ch", type=int, default=1, choices=[0, 1])
    parser.add_argument("--depth_scale", type=float, default=65535.0)

    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--spatial_depth", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--code_seq_len", type=int, default=4)
    parser.add_argument("--z_rgb_feature_dim", type=int, default=4096)

    # Because GT is removed, these must be provided or correctly inferred from checkpoint.
    parser.add_argument("--z_depth_feature_dim", type=int, default=1024)
    parser.add_argument("--predict_token_features", type=int, default=0, choices=[0, 1])

    parser.add_argument("--strict", type=int, default=1, choices=[0, 1])
    parser.add_argument("--check_length_alignment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()

    print("checkpoint:", args.checkpoint)
    print("z_depth_path:", args.z_depth_path)
    print("z_rgb_feature_manifest:", args.z_rgb_feature_manifest)
    print("feature_output_dir:", args.feature_output_dir)
    print("feature_prefix:", args.feature_prefix)
    print("uses_ground_truth: False")

    dataset = Stage25InferenceNoGTDataset(
        z_depth_path=args.z_depth_path,
        z_rgb_feature_manifest=args.z_rgb_feature_manifest,
        image_size=args.image_size,
        repeat_depth_to_3ch=bool(args.repeat_depth_to_3ch),
        depth_scale=args.depth_scale,
        check_length_alignment=bool(args.check_length_alignment),
    )

    print("dataset length:", len(dataset))
    sample = dataset[0]
    print("sample id:", sample["id"])
    print("sample depth1:", sample["depth1"].shape, sample["depth1"].dtype)
    print("sample z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)

    # Load ckpt first only for optional config inference.
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu")
    z_depth_feature_dim, predict_token_features = infer_output_config_from_checkpoint(
        ckpt_raw,
        default_dim=args.z_depth_feature_dim,
        default_predict_token_features=bool(args.predict_token_features),
    )
    print("z_depth_feature_dim:", z_depth_feature_dim)
    print("predict_token_features:", predict_token_features)

    model = LatentActionQuantizationStage25Model4(
        dim=args.dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        spatial_depth=args.spatial_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        code_seq_len=args.code_seq_len,
        z_rgb_feature_dim=args.z_rgb_feature_dim,
        z_depth_feature_dim=z_depth_feature_dim,
        predict_token_features=predict_token_features,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
    ).cuda()

    load_result, _ = load_model4_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        strict=bool(args.strict),
    )
    print("checkpoint load result:", load_result)
    model.eval()

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(**loader_kwargs)

    feature_writer = FeaturePartWriter(
        output_dir=args.feature_output_dir,
        prefix=args.feature_prefix,
        part_size=args.feature_part_size,
    )

    output_jsonl_f = None
    if args.output_jsonl:
        output_jsonl_path = Path(args.output_jsonl)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl_f = output_jsonl_path.open("w", encoding="utf-8")

    total_seen = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            depth1 = batch["depth1"].cuda(non_blocking=True).float()
            z_rgb_features = batch["z_rgb_features"].cuda(non_blocking=True).float()

            z_depth_feature_pred = model.extract_z_depth_feature(
                depth1=depth1,
                z_rgb_features=z_rgb_features,
            )

            z_depth_feature_pred_cpu = z_depth_feature_pred.detach().cpu().float()
            z_rgb_feature_input_cpu = batch["z_rgb_features"].detach().cpu().float() if args.save_rgb else None

            batch_ids = batch.get("id", [str(i) for i in range(z_depth_feature_pred_cpu.shape[0])])
            batch_depth_paths = batch.get("depth1_path", [""] * z_depth_feature_pred_cpu.shape[0])
            bs = z_depth_feature_pred_cpu.shape[0]

            for i in range(bs):
                sample_id = str(batch_ids[i])
                depth1_path = str(batch_depth_paths[i])

                feature_writer.add(
                    sample_id=sample_id,
                    depth1_path=depth1_path,
                    z_depth_feature_pred=z_depth_feature_pred_cpu[i],
                    z_rgb_feature_input=(
                        z_rgb_feature_input_cpu[i] if z_rgb_feature_input_cpu is not None else None
                    ),
                )

                if output_jsonl_f is not None:
                    item = {
                        "id": sample_id,
                        "depth1_path": depth1_path,
                        "z_depth_feature_pred_shape": list(z_depth_feature_pred_cpu[i].shape),
                        "model_name": MODEL_NAME,
                        "stage": STAGE_NAME,
                        "dataset": DATASET_NAME,
                    }
                    if args.save_rgb:
                        item["z_rgb_feature_input_shape"] = list(z_rgb_feature_input_cpu[i].shape)
                    output_jsonl_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            total_seen += bs
            if batch_idx % args.log_every == 0:
                print(
                    f"batch {batch_idx} | samples {total_seen} | "
                    f"z_depth_feature_pred_shape {list(z_depth_feature_pred.shape)}"
                )

    if output_jsonl_f is not None:
        output_jsonl_f.close()

    feature_writer.flush(force=True)
    feature_writer.save_manifest(args=args)

    print("Done.")
    print("total_samples:", feature_writer.total_samples)
    print("num_parts:", len(feature_writer.parts))
    if len(feature_writer.parts) > 0:
        print("first part:", feature_writer.parts[0]["path"])
        print("last part:", feature_writer.parts[-1]["path"])


if __name__ == "__main__":
    main()
