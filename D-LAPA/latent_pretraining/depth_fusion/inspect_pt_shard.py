import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from latent_pretraining.depth_fusion.data_libero import (
    ACTION_CANDIDATES,
    DEPTH_FEATURE_CANDIDATES,
    IMAGE_KEY_CANDIDATES,
    RGB_FEATURE_CANDIDATES,
    discover_part_files,
    load_manifest,
    resolve_feature_keys,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect LIBERO depth-fusion .pt feature shards.")
    parser.add_argument("--pt_file", type=Path, default=None, help="Path to one .pt shard.")
    parser.add_argument("--data_dir", type=Path, default=None, help="Directory containing *_part*.pt shards.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON.")
    parser.add_argument("--sample_count", type=int, default=3)
    parser.add_argument("--rgb_feature_key", type=str, default="auto")
    parser.add_argument("--depth_feature_key", type=str, default="auto")
    parser.add_argument("--action_key", type=str, default="auto")
    parser.add_argument("--image_key", type=str, default="auto")
    return parser.parse_args()


def summarize_value(value: Any) -> Dict[str, Any]:
    if torch.is_tensor(value):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        first_type = type(value[0]).__name__ if value else None
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first_type": first_type,
            "first_value": str(value[0]) if value else None,
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(value.keys()),
        }
    return {
        "type": type(value).__name__,
        "value": str(value),
    }


def value_at(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        item = value[index]
        if item.ndim == 0:
            return item.item()
        return item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def choose_pt_file(pt_file: Optional[Path], data_dir: Optional[Path], manifest: Optional[Path]) -> Path:
    if pt_file is not None:
        return pt_file
    if data_dir is None:
        raise ValueError("Provide either --pt_file or --data_dir.")
    part_files = discover_part_files(data_dir, manifest)
    if not part_files:
        raise ValueError(f"No .pt part files found under {data_dir}.")
    return part_files[0]


def first_present(shard: Dict[str, Any], candidates):
    for key in candidates:
        if key in shard:
            return key
    return None


def manifest_key(manifest: Dict[str, Any], keys):
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve_for_inspection(shard: Dict[str, Any], manifest: Dict[str, Any], args):
    try:
        rgb_key, depth_key, action_key, image_key = resolve_feature_keys(
            shard,
            manifest=manifest,
            rgb_feature_key=args.rgb_feature_key,
            depth_feature_key=args.depth_feature_key,
            action_key=args.action_key,
            image_key=args.image_key,
        )
        return rgb_key, depth_key, action_key, image_key, None
    except KeyError as exc:
        rgb_key = args.rgb_feature_key if args.rgb_feature_key != "auto" else (
            manifest_key(manifest, ("rgb_feature_key_input", "rgb_feature_key"))
            or first_present(shard, RGB_FEATURE_CANDIDATES)
        )
        depth_key = args.depth_feature_key if args.depth_feature_key != "auto" else (
            manifest_key(manifest, ("feature_key", "feature_key_pred", "depth_feature_key"))
            or first_present(shard, DEPTH_FEATURE_CANDIDATES)
        )
        action_key = args.action_key if args.action_key != "auto" else (
            manifest_key(manifest, ("action_key",)) or first_present(shard, ACTION_CANDIDATES)
        )
        image_key = args.image_key if args.image_key != "auto" else (
            manifest_key(manifest, ("image_key", "image_path_key")) or first_present(shard, IMAGE_KEY_CANDIDATES)
        )
        if image_key not in shard:
            image_key = None
        return rgb_key, depth_key, action_key, image_key, str(exc)


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    pt_file = choose_pt_file(args.pt_file, args.data_dir, args.manifest)
    shard = torch.load(pt_file, map_location="cpu")
    if not isinstance(shard, dict):
        raise TypeError(f"Expected {pt_file} to contain a dict, got {type(shard).__name__}.")

    rgb_key, depth_key, action_key, image_key, resolve_error = resolve_for_inspection(shard, manifest, args)

    length_key = action_key if action_key in shard else depth_key
    length = int(shard[length_key].shape[0]) if length_key in shard and torch.is_tensor(shard[length_key]) else 0
    sample_count = min(args.sample_count, length)
    samples = []
    for index in range(sample_count):
        sample = {"index": index}
        if rgb_key in shard:
            sample["rgb_feature_first_values"] = value_at(shard[rgb_key], index)[:5]
        if depth_key in shard:
            sample["depth_feature_first_values"] = value_at(shard[depth_key], index)[:5]
        if action_key in shard:
            sample["action"] = value_at(shard[action_key], index)
        if image_key is not None and image_key in shard:
            sample["image_name"] = value_at(shard[image_key], index)
        samples.append(sample)

    report = {
        "pt_file": str(pt_file),
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "manifest_summary": {
            "prefix": manifest.get("prefix"),
            "total_samples": manifest.get("total_samples"),
            "feature_key": manifest.get("feature_key"),
            "rgb_feature_key_input": manifest.get("rgb_feature_key_input"),
            "num_parts": manifest.get("num_parts"),
        },
        "keys": {key: summarize_value(value) for key, value in sorted(shard.items())},
        "resolved_keys": {
            "rgb_feature_key": rgb_key,
            "depth_feature_key": depth_key,
            "action_key": action_key,
            "image_key": image_key,
            "resolve_error": resolve_error,
        },
        "num_samples_in_shard": length,
        "samples": samples,
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
