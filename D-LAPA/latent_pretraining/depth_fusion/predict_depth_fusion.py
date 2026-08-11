import argparse
import json
from pathlib import Path

import torch

from latent_pretraining.depth_fusion.data_libero import (
    DEPTH_FEATURE_CANDIDATES,
    ID_KEY_CANDIDATES,
    RGB_FEATURE_CANDIDATES,
    ShardFieldIndex,
    _first_present_key,
    _manifest_key,
    _value_at,
    discover_part_files,
    load_manifest,
)
from latent_pretraining.depth_fusion.model import DepthFusionConfig, DepthFusionPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run depth-fusion action prediction from precomputed features.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rgb_data_dir", type=Path, default=None)
    parser.add_argument("--rgb_manifest", type=Path, default=None)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--rgb_feature_key", type=str, default="auto")
    parser.add_argument("--depth_feature_key", type=str, default="auto")
    parser.add_argument("--id_key", type=str, default="auto")
    parser.add_argument("--rgb_id_key", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = DepthFusionConfig(**checkpoint["config"])
    model = DepthFusionPolicy(config)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    manifest = load_manifest(args.manifest)
    part_files = discover_part_files(args.data_dir, args.manifest)
    first_shard = torch.load(part_files[0], map_location="cpu")

    id_key = args.id_key
    if id_key == "auto":
        id_key = (
            _manifest_key(manifest, ("id_key", "image_key", "image_path_key"))
            or _first_present_key(first_shard, ID_KEY_CANDIDATES)
        )
    depth_key = args.depth_feature_key
    if depth_key == "auto":
        depth_key = (
            _manifest_key(manifest, ("feature_key", "feature_key_pred", "depth_feature_key"))
            or _first_present_key(first_shard, DEPTH_FEATURE_CANDIDATES)
        )
    rgb_key = args.rgb_feature_key
    if rgb_key == "auto":
        rgb_key = (
            _manifest_key(manifest, ("rgb_feature_key_input", "rgb_feature_key"))
            or _first_present_key(first_shard, RGB_FEATURE_CANDIDATES)
        )

    rgb_index = None
    if rgb_key not in first_shard:
        if args.rgb_data_dir is None:
            raise KeyError(
                f"Primary shards do not contain RGB key {rgb_key!r}. "
                "Pass --rgb_data_dir/--rgb_manifest."
            )
        rgb_manifest = load_manifest(args.rgb_manifest)
        rgb_part_files = discover_part_files(args.rgb_data_dir, args.rgb_manifest)
        rgb_first = torch.load(rgb_part_files[0], map_location="cpu")
        if rgb_key not in rgb_first:
            rgb_key = (
                _manifest_key(rgb_manifest, ("rgb_feature_key_input", "rgb_feature_key"))
                or _first_present_key(rgb_first, RGB_FEATURE_CANDIDATES)
            )
        rgb_index = ShardFieldIndex(
            rgb_part_files,
            value_key=rgb_key,
            id_key=args.rgb_id_key,
            manifest=rgb_manifest,
            preload=False,
            label="RGB feature",
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output_jsonl.open("w") as fout, torch.no_grad():
        for part_file in part_files:
            shard = torch.load(part_file, map_location="cpu")
            ids = shard[id_key]
            for local_index, sample_id in enumerate(ids):
                sample_id = str(sample_id)
                if rgb_key in shard:
                    rgb_feature = _value_at(shard[rgb_key], local_index).float()
                else:
                    if sample_id not in rgb_index:
                        continue
                    rgb_feature = rgb_index.get(sample_id).float()
                depth_feature = _value_at(shard[depth_key], local_index).float()
                pred = model(
                    rgb_feature.unsqueeze(0).to(args.device),
                    depth_feature.unsqueeze(0).to(args.device),
                )[0]
                fout.write(json.dumps({"id": sample_id, "pred_action": pred.cpu().tolist()}) + "\n")
                written += 1
                if written >= args.max_samples:
                    print(json.dumps({"output_jsonl": str(args.output_jsonl), "num_predictions": written}))
                    return
    print(json.dumps({"output_jsonl": str(args.output_jsonl), "num_predictions": written}))


if __name__ == "__main__":
    main()
