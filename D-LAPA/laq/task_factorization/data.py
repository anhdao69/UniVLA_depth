from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


FRAME_NUMBER = re.compile(r"(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected an annotation list in {path}")
    result: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict) or "id" not in row or "label" not in row:
            raise ValueError("each annotation must contain id and label")
        key = str(row["id"])
        if key in result:
            raise ValueError(f"duplicate annotation id {key}")
        result[key] = row
    return result


def fill_instruction(template: str, placeholders: Sequence[str]) -> str:
    text = str(template)
    for value in placeholders:
        text = text.replace("[something]", str(value), 1)
    return text


def frame_number(path: Path) -> int:
    match = FRAME_NUMBER.search(path.stem)
    if match is None:
        raise ValueError(f"frame filename has no numeric suffix: {path.name}")
    return int(match.group(1))


def read_depth(path: str | Path, image_size: int) -> Tensor:
    path = Path(path)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"failed to read depth image {path}")
    if depth.dtype != np.uint16:
        raise TypeError(f"expected uint16 depth at {path}, got {depth.dtype}")
    if depth.ndim != 2:
        raise ValueError(f"expected one-channel depth at {path}, got shape {depth.shape}")
    depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    normalized = depth.astype(np.float32) / np.float32(65535.0)
    return torch.from_numpy(normalized).unsqueeze(0)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def _stratified_video_split(
    videos: Dict[str, Dict[str, Any]], val_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for video_id, record in videos.items():
        groups[str(record["template"])].append(video_id)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    rng = random.Random(seed)
    for template in sorted(groups):
        ids = sorted(groups[template])
        rng.shuffle(ids)
        if len(ids) <= 1:
            train_ids.update(ids)
            continue
        count = max(1, int(round(len(ids) * val_fraction)))
        count = min(count, len(ids) - 1)
        val_ids.update(ids[:count])
        train_ids.update(ids[count:])
    return train_ids, val_ids


def build_pair_manifests(
    depth_root: Path,
    annotation_path: Path,
    output_dir: Path,
    *,
    delta: int = 30,
    val_fraction: float = 0.05,
    seed: int = 42,
    max_videos: Optional[int] = None,
    verify_all: bool = False,
) -> Dict[str, Any]:
    depth_root = depth_root.resolve()
    annotation_path = annotation_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(annotation_path)
    video_dirs = sorted(
        (path for path in depth_root.iterdir() if path.is_dir()),
        key=lambda value: (not value.name.isdigit(), int(value.name) if value.name.isdigit() else value.name),
    )
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]

    videos: Dict[str, Dict[str, Any]] = {}
    total_frames = 0
    verified_frames = 0
    for video_dir in video_dirs:
        video_id = video_dir.name
        if video_id not in annotations:
            raise KeyError(f"missing annotation for depth video {video_id}")
        frame_paths = sorted(video_dir.glob("*.png"), key=frame_number)
        if not frame_paths:
            raise ValueError(f"no PNG frames in {video_dir}")
        indexed: Dict[int, Path] = {}
        for path in frame_paths:
            index = frame_number(path)
            if index in indexed:
                raise ValueError(f"duplicate frame number {index} in {video_dir}")
            indexed[index] = path.resolve()
        total_frames += len(indexed)

        paths_to_verify = list(indexed.values()) if verify_all else [indexed[min(indexed)], indexed[max(indexed)]]
        for path in dict.fromkeys(paths_to_verify):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
                dtype = None if raw is None else raw.dtype
                shape = None if raw is None else raw.shape
                raise ValueError(f"unverified depth encoding at {path}: dtype={dtype}, shape={shape}")
            verified_frames += 1

        valid_starts = [index for index in sorted(indexed) if index + delta in indexed]
        if not valid_starts:
            continue
        ann = annotations[video_id]
        template = str(ann.get("template", ""))
        placeholders = [str(value) for value in ann.get("placeholders", [])]
        videos[video_id] = {
            "video_id": video_id,
            "label": str(ann["label"]),
            "template": template,
            "placeholders": placeholders,
            "filled_instruction": fill_instruction(template, placeholders),
            "indexed": indexed,
            "starts": valid_starts,
        }

    train_ids, val_ids = _stratified_video_split(videos, val_fraction, seed)
    if train_ids & val_ids:
        raise RuntimeError("video leakage between train and validation splits")

    def rows(ids: set[str]):
        seen: set[str] = set()
        for video_id in sorted(ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
            item = videos[video_id]
            for start in item["starts"]:
                future = start + delta
                pair_id = f"{video_id}__t{start:06d}__tp{future:06d}"
                if pair_id in seen:
                    raise RuntimeError(f"duplicate pair id {pair_id}")
                seen.add(pair_id)
                yield {
                    "id": pair_id,
                    "pair_id": pair_id,
                    "video_id": video_id,
                    "t": start,
                    "future_t": future,
                    "depth_t_path": str(item["indexed"][start]),
                    "depth_future_path": str(item["indexed"][future]),
                    "label": item["label"],
                    "template": item["template"],
                    "placeholders": item["placeholders"],
                    "filled_instruction": item["filled_instruction"],
                }

    train_path = output_dir / "train_pairs.jsonl"
    val_path = output_dir / "val_pairs.jsonl"
    train_pairs = _write_jsonl(train_path, rows(train_ids))
    val_pairs = _write_jsonl(val_path, rows(val_ids))
    metadata = {
        "schema_version": 1,
        "depth_root": str(depth_root),
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "delta": delta,
        "seed": seed,
        "val_fraction": val_fraction,
        "source_videos": len(video_dirs),
        "usable_videos": len(videos),
        "train_videos": len(train_ids),
        "val_videos": len(val_ids),
        "total_frames": total_frames,
        "verified_frames": verified_frames,
        "verify_all": verify_all,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_manifest": train_path.name,
        "val_manifest": val_path.name,
    }
    metadata["manifest_hash"] = stable_json_hash(metadata)
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if "id" not in row:
                    raise ValueError(f"missing id at {path}:{line_number}")
                rows.append(row)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate pair IDs in {path}")
    return rows


class TextEmbeddingCache:
    def __init__(self, path: Path) -> None:
        package = torch.load(path, map_location="cpu", weights_only=False)
        self.metadata = package["metadata"]
        self.entries: Dict[str, Tensor] = package["entries"]

    def get(self, text: str) -> Tensor:
        try:
            return self.entries[text].float()
        except KeyError as error:
            raise KeyError(f"instruction absent from text cache: {text!r}") from error


class DepthPairDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        image_size: int,
        text_cache_path: Optional[Path] = None,
    ) -> None:
        self.rows = load_jsonl(manifest_path)
        self.image_size = int(image_size)
        self.text_cache = TextEmbeddingCache(text_cache_path) if text_cache_path else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        result: Dict[str, Any] = {
            "id": str(row["id"]),
            "row": row,
            "depth_pair": torch.stack(
                [
                    read_depth(row["depth_t_path"], self.image_size),
                    read_depth(row["depth_future_path"], self.image_size),
                ],
                dim=0,
            ),
        }
        if self.text_cache is not None:
            result["language"] = self.text_cache.get(str(row["label"]))
        return result


def collate_depth_pairs(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "id": [item["id"] for item in batch],
        "row": [item["row"] for item in batch],
        "depth_pair": torch.stack([item["depth_pair"] for item in batch]),
    }
    if "language" in batch[0]:
        lengths = [item["language"].shape[0] for item in batch]
        max_length = max(lengths)
        dim = batch[0]["language"].shape[-1]
        language = torch.zeros(len(batch), max_length, dim, dtype=torch.float32)
        mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
        for index, item in enumerate(batch):
            length = lengths[index]
            language[index, :length] = item["language"]
            mask[index, :length] = True
        result["language"] = language
        result["language_mask"] = mask
    return result
