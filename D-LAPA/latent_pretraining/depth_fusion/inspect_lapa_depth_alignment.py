import argparse
import json
from pathlib import Path

import torch

from latent_pretraining.depth_fusion.data_libero import (
    DEPTH_FEATURE_CANDIDATES,
    ShardFieldIndex,
    _first_present_key,
    _manifest_key,
    discover_part_files,
    load_manifest,
)
from latent_pretraining.depth_fusion.id_mapping import resolve_lapa_sample_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect whether LAPA JSONL samples match offline Stage-2.5 depth features by id."
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="LAPA fine-tuning JSONL.")
    parser.add_argument("--depth_data_dir", type=Path, required=True, help="Directory containing depth .pt/.pth parts.")
    parser.add_argument("--depth_manifest", type=Path, default=None, help="Optional depth manifest JSON.")
    parser.add_argument("--json_id_key", type=str, default="id", help="ID field in the LAPA JSONL.")
    parser.add_argument(
        "--json_id_source",
        type=str,
        default="auto",
        choices=("auto", "id", "image"),
        help="Use JSON id field, derive from image path, or try id then image.",
    )
    parser.add_argument("--depth_id_key", type=str, default="auto", help="ID field in the depth shards.")
    parser.add_argument("--depth_feature_key", type=str, default="auto", help="Feature key in the depth shards.")
    parser.add_argument("--sample_count", type=int, default=5, help="Number of matched/missing examples to print.")
    return parser.parse_args()


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            yield line_number, item


def safe_resolve_sample_id(item, args):
    try:
        return resolve_lapa_sample_id(item, id_key=args.json_id_key, source=args.json_id_source), None
    except ValueError as exc:
        return None, str(exc)


def main():
    args = parse_args()
    manifest = load_manifest(args.depth_manifest)
    part_files = discover_part_files(args.depth_data_dir, args.depth_manifest)
    if not part_files:
        raise FileNotFoundError(f"No depth feature parts found under {args.depth_data_dir}")

    first_shard = torch.load(part_files[0], map_location="cpu")
    depth_feature_key = args.depth_feature_key
    if depth_feature_key == "auto":
        depth_feature_key = (
            _manifest_key(manifest, ("feature_key", "feature_key_pred", "depth_feature_key"))
            or _first_present_key(first_shard, DEPTH_FEATURE_CANDIDATES)
        )
    if depth_feature_key not in first_shard:
        raise KeyError(
            f"Depth feature key {depth_feature_key!r} was not found. "
            f"Available keys: {sorted(first_shard.keys())}"
        )

    depth_index = ShardFieldIndex(
        part_files,
        value_key=depth_feature_key,
        id_key=args.depth_id_key,
        manifest=manifest,
        preload=False,
        label="offline depth feature",
    )

    matched = []
    missing = []
    total = 0
    missing_json_id = 0
    for line_number, item in iter_jsonl(args.jsonl):
        total += 1
        sample_id, resolve_error = safe_resolve_sample_id(item, args)
        if sample_id is None:
            missing_json_id += 1
            if len(missing) < args.sample_count:
                missing.append(
                    {
                        "line": line_number,
                        "reason": f"could_not_resolve_json_id:{args.json_id_source}:{args.json_id_key}",
                        "available_keys": sorted(item.keys()),
                        "image": item.get("image"),
                        "resolve_error": resolve_error,
                    }
                )
            continue
        sample_id = str(sample_id)
        if sample_id in depth_index:
            if len(matched) < args.sample_count:
                depth_feature = depth_index.get(sample_id)
                matched.append(
                    {
                        "line": line_number,
                        "id": sample_id,
                        "image": item.get("image"),
                        "instruction": item.get("instruction"),
                        "depth_shape": list(depth_feature.shape) if hasattr(depth_feature, "shape") else None,
                        "depth_first_values": depth_feature.flatten()[:5].tolist()
                        if torch.is_tensor(depth_feature)
                        else None,
                    }
                )
        elif len(missing) < args.sample_count:
            missing.append(
                {
                    "line": line_number,
                    "id": sample_id,
                    "image": item.get("image"),
                    "instruction": item.get("instruction"),
                    "reason": "id_not_found_in_depth_shards",
                }
            )

    matched_count = sum(
        1
        for _, item in iter_jsonl(args.jsonl)
        if safe_resolve_sample_id(item, args)[0] in depth_index
    )
    result = {
        "depth": {
            "data_dir": str(args.depth_data_dir),
            "manifest": str(args.depth_manifest) if args.depth_manifest else None,
            "parts": len(part_files),
            "first_part": str(part_files[0]),
            "available_keys": sorted(first_shard.keys()),
            "resolved_id_key": depth_index.id_key,
            "resolved_feature_key": depth_feature_key,
            "depth_index_size": len(depth_index),
        },
        "jsonl": {
            "path": str(args.jsonl),
            "json_id_key": args.json_id_key,
            "json_id_source": args.json_id_source,
            "total_rows": total,
            "missing_json_id": missing_json_id,
            "matched_rows": matched_count,
            "unmatched_rows": total - matched_count - missing_json_id,
            "match_rate": matched_count / max(1, total - missing_json_id),
        },
        "matched_examples": matched,
        "missing_examples": missing,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
