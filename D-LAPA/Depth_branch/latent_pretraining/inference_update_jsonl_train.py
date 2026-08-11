import os
import json
import argparse
import time
import re
from typing import Optional, Dict, List

import numpy as np
from PIL import Image
import torch

from latent_pretraining.sampler_latent_pretrain import DeltaSampler
from latent_pretraining.delta_llama import VideoLLaMAConfig
from tux import JaxDistributedConfig, set_random_seed


class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)


class FeatureShardWriter:
    def __init__(self, output_dir, prefix, flush_every=8192):
        self.output_dir = output_dir
        self.prefix = prefix
        self.flush_every = flush_every

        os.makedirs(self.output_dir, exist_ok=True)

        self.part_idx = 0
        self.total = 0
        self.manifest = []

        self.ids = []
        self.video_ids = []
        self.image_paths = []
        self.instructions = []
        self.z_rgb_indices = []
        self.z_rgb_features = []

    def add(
        self,
        sample_id,
        video_id,
        image_path,
        instruction,
        z_rgb_indices,
        z_rgb_feature,
    ):
        z_rgb_feature = np.asarray(z_rgb_feature)

        self.ids.append(str(sample_id))
        self.video_ids.append(str(video_id))
        self.image_paths.append(str(image_path))
        self.instructions.append(str(instruction))
        self.z_rgb_indices.append([int(x) for x in z_rgb_indices])
        self.z_rgb_features.append(z_rgb_feature.astype(np.float16))

        if len(self.ids) >= self.flush_every:
            self.flush()

    def flush(self):
        if len(self.ids) == 0:
            return

        features = np.stack(self.z_rgb_features, axis=0)
        indices = np.asarray(self.z_rgb_indices, dtype=np.int64)

        part_path = os.path.join(
            self.output_dir,
            f"{self.prefix}_part{self.part_idx:05d}.pt",
        )

        pkg = {
            "ids": list(self.ids),
            "video_ids": list(self.video_ids),
            "image_paths": list(self.image_paths),
            "instructions": list(self.instructions),
            "z_rgb_indices": torch.from_numpy(indices).long(),
            "z_rgb_features": torch.from_numpy(features).half(),
        }

        torch.save(pkg, part_path)

        self.manifest.append({
            "part": self.part_idx,
            "path": part_path,
            "num_samples": len(self.ids),
            "z_rgb_features_shape": list(pkg["z_rgb_features"].shape),
            "z_rgb_indices_shape": list(pkg["z_rgb_indices"].shape),
        })

        self.total += len(self.ids)

        print(
            f"[FeatureShardWriter] saved {part_path} | "
            f"samples={len(self.ids)} | "
            f"features={list(pkg['z_rgb_features'].shape)}"
        )

        self.part_idx += 1

        self.ids.clear()
        self.video_ids.clear()
        self.image_paths.clear()
        self.instructions.clear()
        self.z_rgb_indices.clear()
        self.z_rgb_features.clear()

    def close(self):
        self.flush()

        manifest_path = os.path.join(
            self.output_dir,
            f"{self.prefix}_manifest.json",
        )

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prefix": self.prefix,
                    "total_samples": self.total,
                    "num_parts": self.part_idx,
                    "parts": self.manifest,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[FeatureShardWriter] wrote manifest: {manifest_path}")


class LAPAInference:
    def __init__(self, image_size: int = 256, **kwargs) -> None:
        flags = FLAGSClass(kwargs)
        self.model = DeltaSampler(FLAGS=flags)
        self.image_size = image_size
        self.tokens_per_delta = kwargs["tokens_per_delta"]

    def inference(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        return_feature: bool = False,
        feature_pool: str = "mean",
    ):
        assert image.dtype == np.uint8

        image_pil = Image.fromarray(image)
        prompts = [{"image": [image_pil], "question": task_description}]

        if return_feature:
            output = self.model(
                prompts,
                return_feature_before_head=True,
                feature_pool=feature_pool,
            )

            latent_action = output["latent_action"][0]
            z_rgb_feature = output["z_rgb_feature_before_head"][0]

            return latent_action, z_rgb_feature

        latent_output = self.model(prompts)
        latent_action = latent_output[0]

        return latent_action


def load_all_labels(
    label_root: str,
    label_json: Optional[str] = None,
    include_validation_labels: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Load label/instruction metadata.

    If label_json is provided:
        Load only that JSON file.

    Otherwise:
        Load label_root/train.json.
        Optionally load label_root/validation.json.

    Return:
        {
            "video_id": {
                "instruction": "...",
                "split": "train"
            }
        }
    """

    id_to_meta = {}

    if label_json is not None:
        split_files = {
            "train": label_json,
        }
    else:
        split_files = {
            "train": os.path.join(label_root, "train.json"),
        }

        if include_validation_labels:
            split_files["validation"] = os.path.join(label_root, "validation.json")

    for split_name, path in split_files.items():
        if not os.path.exists(path):
            print(f"[Warn] label file not found: {path}")
            continue

        print(f"[Info] loading label file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            vid = str(item["id"])
            instruction = item["label"]

            id_to_meta[vid] = {
                "instruction": instruction,
                "split": split_name,
            }

    return id_to_meta


def natural_key(filename: str):
    """
    Sort frames numerically when filenames contain frame indices.
    Example:
        frame_2.jpg comes before frame_10.jpg
    """

    stem = os.path.splitext(filename)[0]
    m = re.search(r"(\d+)", stem)

    if m:
        return int(m.group(1))

    return stem


def sorted_frame_files(frame_dir: str) -> List[str]:
    files = [
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    files.sort(key=natural_key)
    return files


def load_rgb_image(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.uint8)


def load_folder_list(folder_list: Optional[str]) -> Optional[List[str]]:
    if folder_list is None:
        return None

    with open(folder_list, "r", encoding="utf-8") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    return video_ids


def read_existing_ids(jsonl_path: str) -> set:
    existing = set()

    if not os.path.exists(jsonl_path):
        return existing

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "id" in item:
                    existing.add(str(item["id"]))
            except Exception:
                continue

    return existing


def save_debug_json(
    save_path: str,
    video_id: str,
    instruction: str,
    split: str,
    frame_files: List[str],
    z_rgb_indices: List[List[int]],
):
    data = {
        "video_id": video_id,
        "instruction": instruction,
        "split": split,
        "data": [],
    }

    for f, z in zip(frame_files, z_rgb_indices):
        data["data"].append({
            "frame": f,
            "z_rgb": z,
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_output_paths(args):
    """
    Output structure:

    feature_dir / feature_prefix /
    ├── feature_prefix.jsonl
    ├── feature_prefix_part00000.pt
    └── feature_prefix_manifest.json

    Example:

    --feature_dir /datasets/libero_corl/features/libero_goal
    --feature_prefix z_rgb_train_shard0

    gives:

    /datasets/libero_corl/features/libero_goal/z_rgb_train_shard0/
    ├── z_rgb_train_shard0.jsonl
    ├── z_rgb_train_shard0_part00000.pt
    └── z_rgb_train_shard0_manifest.json
    """

    feature_output_dir = os.path.join(args.feature_dir, args.feature_prefix)
    os.makedirs(feature_output_dir, exist_ok=True)

    if args.unshuffled_jsonl is None:
        unshuffled_jsonl = os.path.join(
            feature_output_dir,
            f"{args.feature_prefix}.jsonl",
        )
    else:
        unshuffled_jsonl = args.unshuffled_jsonl

    feature_prefix = args.feature_prefix

    return feature_output_dir, feature_prefix, unshuffled_jsonl


def main():
    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="something-something-v2",
    )
    parser.add_argument(
        "--frames_dirname",
        type=str,
        default="frames_10",
    )
    parser.add_argument(
        "--labels_dirname",
        type=str,
        default="labels",
    )
    parser.add_argument(
        "--label_json",
        type=str,
        default=None,
        help="Path to a specific label JSON file. If set, this overrides labels_dirname/train.json",
    )
    parser.add_argument(
        "--folder_list",
        type=str,
        default=None,
        help="TXT file containing video folder IDs to process",
    )

    # Output paths
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Root folder to save feature shard folders, e.g. /datasets/libero_corl/features/libero_goal",
    )
    parser.add_argument(
        "--feature_prefix",
        type=str,
        required=True,
        help="Shard output name, e.g. z_rgb_train_shard0",
    )
    parser.add_argument(
        "--unshuffled_jsonl",
        type=str,
        default=None,
        help="Optional full JSONL output path. If not set, use feature_dir/feature_prefix/feature_prefix.jsonl",
    )
    parser.add_argument(
        "--feature_flush_every",
        type=int,
        default=8192,
        help="Number of samples per .pt part file",
    )

    # Debug
    parser.add_argument(
        "--debug_dirname",
        type=str,
        default="z_rgb_indices_stage2_debug",
    )
    parser.add_argument(
        "--save_debug_json",
        action="store_true",
    )
    parser.add_argument(
        "--debug_max_videos",
        type=int,
        default=20,
    )

    # Processing options
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output JSONL instead of overwriting",
    )
    parser.add_argument(
        "--include_validation_labels",
        action="store_true",
        help="Also load labels/validation.json if it exists. Ignored when --label_json is set.",
    )

    # LAPA / model arguments
    parser.add_argument(
        "--tokens_per_delta",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--vqgan_checkpoint",
        type=str,
        default="lapa_checkpoints/vqgan",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="lapa_checkpoints/tokenizer.model",
    )
    parser.add_argument(
        "--multi_image",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--jax_distributed",
        type=dict,
        default=JaxDistributedConfig.get_default_config(),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--mesh_dim",
        type=str,
        default="1,1,1,1",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
    )
    parser.add_argument(
        "--load_llama_config",
        type=str,
        default="7b",
    )
    parser.add_argument(
        "--update_llama_config",
        type=str,
        default=(
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
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default="params::lapa_checkpoints/streaming_params_22485",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    # Resolve paths
    dataset_root = args.dataset_root
    frames_root = os.path.join(dataset_root, args.frames_dirname)
    labels_root = os.path.join(dataset_root, args.labels_dirname)
    debug_root = os.path.join(dataset_root, args.debug_dirname)

    feature_output_dir, feature_prefix, unshuffled_jsonl = resolve_output_paths(args)

    print("dataset_root:", dataset_root)
    print("frames_root:", frames_root)
    print("labels_root:", labels_root)
    print("label_json:", args.label_json)
    print("feature_output_dir:", feature_output_dir)
    print("feature_prefix:", feature_prefix)
    print("unshuffled_jsonl:", unshuffled_jsonl)

    # Prepare output dirs
    parent_dir = os.path.dirname(unshuffled_jsonl)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if args.save_debug_json:
        os.makedirs(debug_root, exist_ok=True)

    # Initialize model configs
    args.tokenizer = VideoLLaMAConfig.get_tokenizer_config()
    args.llama = VideoLLaMAConfig.get_default_config()
    args.tokenizer.vocab_file = args.vocab_file

    JaxDistributedConfig.initialize(args.jax_distributed)
    set_random_seed(args.seed)

    lapa = LAPAInference(
        image_size=args.image_size,
        tokens_per_delta=args.tokens_per_delta,
        vqgan_checkpoint=args.vqgan_checkpoint,
        vocab_file=args.vocab_file,
        multi_image=args.multi_image,
        jax_distributed=args.jax_distributed,
        seed=args.seed,
        mesh_dim=args.mesh_dim,
        dtype=args.dtype,
        load_llama_config=args.load_llama_config,
        update_llama_config=args.update_llama_config,
        load_checkpoint=args.load_checkpoint,
        tokenizer=args.tokenizer,
        llama=args.llama,
    )

    feature_writer = FeatureShardWriter(
        output_dir=feature_output_dir,
        prefix=feature_prefix,
        flush_every=args.feature_flush_every,
    )

    # Load labels
    id_to_meta = load_all_labels(
        labels_root,
        label_json=args.label_json,
        include_validation_labels=args.include_validation_labels,
    )

    print(f"loaded_labels: {len(id_to_meta)}")

    if not os.path.isdir(frames_root):
        raise RuntimeError(f"frames_root does not exist: {frames_root}")

    # Load folders to process
    listed_video_ids = load_folder_list(args.folder_list)

    if listed_video_ids is not None:
        video_ids = [
            str(v)
            for v in listed_video_ids
            if os.path.isdir(os.path.join(frames_root, str(v)))
        ]

        missing_video_ids = [
            str(v)
            for v in listed_video_ids
            if not os.path.isdir(os.path.join(frames_root, str(v)))
        ]

        if missing_video_ids:
            print(
                f"[Warn] {len(missing_video_ids)} ids in folder_list are missing under frames_root. "
                f"Examples: {missing_video_ids[:5]}"
            )
    else:
        video_ids = [
            d for d in os.listdir(frames_root)
            if os.path.isdir(os.path.join(frames_root, d))
        ]
        video_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    if args.max_videos is not None:
        video_ids = video_ids[:args.max_videos]

    print(f"videos_to_process: {len(video_ids)}")
    if video_ids:
        print(f"first_video: {video_ids[0]} | last_video: {video_ids[-1]}")

    existing_ids = read_existing_ids(unshuffled_jsonl) if args.skip_existing else set()
    if args.skip_existing:
        print(f"skip_existing=True | existing samples in output: {len(existing_ids)}")

    debug_count = 0
    total_written = 0
    open_mode = "a" if args.append else "w"

    try:
        with open(unshuffled_jsonl, open_mode, encoding="utf-8") as fout:
            for idx, video_id in enumerate(video_ids):
                if video_id not in id_to_meta:
                    print(f"[Skip] video_id={video_id} not found in label files")
                    continue

                frame_dir = os.path.join(frames_root, video_id)
                frame_files = sorted_frame_files(frame_dir)

                if len(frame_files) == 0:
                    print(f"[Skip] video_id={video_id} has no frames")
                    continue

                instruction = id_to_meta[video_id]["instruction"]
                split = id_to_meta[video_id]["split"]

                z_rgb_indices = []
                video_written = 0
                t_start = time.time()

                for frame_name in frame_files:
                    sample_id = f"{video_id}_{os.path.splitext(frame_name)[0]}"

                    if args.skip_existing and sample_id in existing_ids:
                        continue

                    frame_path = os.path.join(frame_dir, frame_name)
                    abs_frame_path = os.path.abspath(frame_path)

                    image = load_rgb_image(abs_frame_path, args.image_size)

                    latent_action, z_rgb_feature = lapa.inference(
                        image,
                        instruction,
                        return_feature=True,
                        feature_pool="mean",
                    )

                    if isinstance(latent_action, np.ndarray):
                        latent_ids = latent_action.astype(np.int16).tolist()
                    else:
                        latent_ids = np.array(latent_action, dtype=np.int16).tolist()

                    if len(latent_ids) == 1 and isinstance(latent_ids[0], list):
                        latent_ids = latent_ids[0]

                    feature_writer.add(
                        sample_id=sample_id,
                        video_id=video_id,
                        image_path=abs_frame_path,
                        instruction=instruction,
                        z_rgb_indices=latent_ids,
                        z_rgb_feature=z_rgb_feature,
                    )

                    elem_dict = {
                        "id": sample_id,
                        "video_id": video_id,
                        "image": abs_frame_path,
                        "delta": [str(i) for i in latent_ids],
                        "instruction": instruction,
                        "vision": [],
                        "fields": "[instruction],[vision],delta",
                    }

                    fout.write(json.dumps(elem_dict, ensure_ascii=False) + "\n")
                    fout.flush()

                    existing_ids.add(sample_id)
                    z_rgb_indices.append(latent_ids)

                    video_written += 1
                    total_written += 1

                if args.save_debug_json and debug_count < args.debug_max_videos:
                    json_path = os.path.join(debug_root, f"{video_id}.json")
                    save_debug_json(
                        json_path,
                        video_id=video_id,
                        instruction=instruction,
                        split=split,
                        frame_files=frame_files,
                        z_rgb_indices=z_rgb_indices,
                    )
                    debug_count += 1

                print(
                    f"[{idx + 1}/{len(video_ids)}] "
                    f"Wrote video_id={video_id} | "
                    f"split={split} | "
                    f"frames={video_written} | "
                    f"time={time.time() - t_start:.2f}s"
                )

    finally:
        feature_writer.close()

    print(f"Done. Wrote {total_written} new lines to: {unshuffled_jsonl}")


if __name__ == "__main__":
    main()