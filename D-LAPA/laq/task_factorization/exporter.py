from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from .data import DepthPairDataset, collate_depth_pairs, sha256_file
from .trainer import load_task_checkpoint


class PartWriter:
    def __init__(self, output_dir: Path, part_size: int, metadata: Dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.part_size = int(part_size)
        self.metadata = metadata
        self.buffer: List[Dict[str, Any]] = []
        self.parts: List[Dict[str, Any]] = []
        self.total = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, row: Dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.part_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        index = len(self.parts)
        path = self.output_dir / f"task_depth_part{index:05d}.pt"
        package = {
            "id": [row["id"] for row in self.buffer],
            "pair_id": [row["id"] for row in self.buffer],
            "video_id": [row["video_id"] for row in self.buffer],
            "t": torch.tensor([row["t"] for row in self.buffer], dtype=torch.long),
            "future_t": torch.tensor([row["future_t"] for row in self.buffer], dtype=torch.long),
            "depth_t_path": [row["depth_t_path"] for row in self.buffer],
            "depth_future_path": [row["depth_future_path"] for row in self.buffer],
            "z_depth_feature_gt": torch.stack([row["task_feature"] for row in self.buffer]).float(),
            "z_depth_indices_gt": torch.stack([row["task_indices"] for row in self.buffer]).long(),
            "z_depth_feature_irr": torch.stack([row["irr_feature"] for row in self.buffer]).float(),
            "z_depth_indices_irr": torch.stack([row["irr_indices"] for row in self.buffer]).long(),
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(package, tmp)
        tmp.replace(path)
        count = len(self.buffer)
        self.parts.append({"path": path.name, "num_samples": count})
        self.total += count
        self.buffer.clear()

    def finish(self) -> Path:
        self.flush()
        manifest = dict(self.metadata)
        manifest.update(
            {
                "schema_version": 1,
                "id_key": "id",
                "feature_key": "z_depth_feature_gt",
                "indices_key": "z_depth_indices_gt",
                "irr_feature_key": "z_depth_feature_irr",
                "irr_indices_key": "z_depth_indices_irr",
                "total_samples": self.total,
                "parts": self.parts,
            }
        )
        path = self.output_dir / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
        return path


@torch.no_grad()
def export_features(
    checkpoint: Path,
    pair_manifest: Path,
    output_dir: Path,
    *,
    batch_size: int = 32,
    num_workers: int = 8,
    part_size: int = 4096,
    max_batches: int | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_task_checkpoint(checkpoint, device)
    if model.config.stage != "task":
        raise ValueError("export requires a task-stage checkpoint")
    dataset = DepthPairDataset(pair_manifest, image_size=model.config.image_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=collate_depth_pairs,
    )
    writer = PartWriter(
        output_dir,
        part_size,
        {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "pair_manifest": str(pair_manifest.resolve()),
            "pair_manifest_sha256": sha256_file(pair_manifest),
        },
    )
    seen: set[str] = set()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        depth = batch["depth_pair"].to(device, non_blocking=True)
        outputs = model.extract_teacher(depth[:, 0], depth[:, 1])
        for index, row in enumerate(batch["row"]):
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ValueError(f"duplicate export ID {sample_id}")
            seen.add(sample_id)
            writer.add(
                {
                    **row,
                    "task_feature": outputs["task_feature"][index].cpu(),
                    "task_indices": outputs["task_indices"][index].cpu(),
                    "irr_feature": outputs["irr_feature"][index].cpu(),
                    "irr_indices": outputs["irr_indices"][index].cpu(),
                }
            )
    return writer.finish()
