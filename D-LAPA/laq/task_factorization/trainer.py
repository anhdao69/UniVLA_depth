from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import ModelConfig
from .data import DepthPairDataset, collate_depth_pairs, sha256_file
from .model import DepthFactorizedLAM


@dataclass
class TrainConfig:
    manifest_path: str
    val_manifest_path: str
    output_dir: str
    text_cache_path: Optional[str] = None
    stage1_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    steps: int = 25_000
    batch_size: int = 64
    grad_accumulation: int = 1
    num_workers: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 0.1
    seed: int = 42
    log_every: int = 10
    save_every: int = 2_500
    eval_every: int = 1_000
    restart_every: int = 1_000
    precision: str = "bf16"
    allow_nonproduction_batch: bool = False


def distributed_info() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return world_size, rank, local_rank


def seed_everything(seed: int, rank: int = 0) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def atomic_torch_save(package: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(package, tmp)
    tmp.replace(path)


def checkpoint_package(
    model: DepthFactorizedLAM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    transfer_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": model_config.stage,
        "step": step,
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "manifest_sha256": sha256_file(Path(train_config.manifest_path)),
        "val_manifest_sha256": sha256_file(Path(train_config.val_manifest_path)),
        "text_cache_sha256": sha256_file(Path(train_config.text_cache_path)) if train_config.text_cache_path else None,
        "parent_checkpoint_sha256": sha256_file(Path(train_config.stage1_checkpoint)) if train_config.stage1_checkpoint else None,
        "transfer_report": transfer_report,
    }


def create_model(
    model_config: ModelConfig, train_config: TrainConfig, device: torch.device
) -> tuple[DepthFactorizedLAM, Optional[Dict[str, Any]]]:
    model = DepthFactorizedLAM(model_config)
    transfer_report = None
    if model_config.stage == "task":
        if not train_config.stage1_checkpoint:
            raise ValueError("task stage requires --stage1-checkpoint")
        parent = torch.load(train_config.stage1_checkpoint, map_location="cpu", weights_only=False)
        if parent.get("stage") != "irr":
            raise ValueError("parent checkpoint is not a Stage-1a checkpoint")
        parent_config = ModelConfig.from_dict(parent["model_config"])
        shared_fields = (
            "image_size", "patch_size", "dim", "encoder_depth", "decoder_depth",
            "heads", "mlp_ratio", "dropout", "num_queries", "vq_dim",
            "irr_codebook_size", "text_dim", "commitment_beta",
        )
        mismatches = {
            field: (getattr(parent_config, field), getattr(model_config, field))
            for field in shared_fields
            if getattr(parent_config, field) != getattr(model_config, field)
        }
        if mismatches:
            raise ValueError(f"Stage-1a/1b architecture mismatch: {mismatches}")
        transfer_report = model.initialize_task_from_stage1(parent["model"])
    return model.to(device), transfer_report


@torch.no_grad()
def validate(model, loader: DataLoader, device: torch.device, max_batches: int = 8) -> Dict[str, float]:
    base = unwrap(model)
    was_training = base.training
    base.eval()
    totals: Dict[str, float] = {"loss": 0.0, "reconstruction_loss": 0.0}
    count = 0
    cfg = base.config
    irr_hist = torch.zeros(cfg.irr_codebook_size, dtype=torch.long)
    task_hist = torch.zeros(cfg.task_codebook_size, dtype=torch.long)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        depth = batch["depth_pair"].to(device, non_blocking=True)
        language = batch.get("language")
        language_mask = batch.get("language_mask")
        if language is not None:
            language = language.to(device, non_blocking=True)
            language_mask = language_mask.to(device, non_blocking=True)
        outputs = base(depth, language, language_mask)
        for key in totals:
            totals[key] += float(outputs[key].detach())
        irr_hist += torch.bincount(outputs["irr_indices"].cpu().reshape(-1), minlength=cfg.irr_codebook_size)
        if cfg.stage == "task":
            task_hist += torch.bincount(outputs["task_indices"].cpu().reshape(-1), minlength=cfg.task_codebook_size)
        count += 1
    if was_training:
        base.train()
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    for name, histogram in (("irr", irr_hist), ("task", task_hist)):
        if name == "task" and cfg.stage != "task":
            continue
        probabilities = histogram.float() / histogram.sum().clamp_min(1)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        normalizer = torch.log(torch.tensor(float(histogram.numel()))).clamp_min(1e-8)
        metrics[f"{name}_active_fraction"] = float((histogram > 0).float().mean())
        metrics[f"{name}_normalized_entropy"] = float(entropy / normalizer)
    return metrics


def train(model_config: ModelConfig, train_config: TrainConfig) -> Path:
    world_size, rank, local_rank = distributed_info()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(train_config.seed, rank)
    effective_batch = world_size * train_config.batch_size * train_config.grad_accumulation
    if effective_batch != 64 and not train_config.allow_nonproduction_batch:
        raise ValueError(
            "production global batch must be 64; use --allow-nonproduction-batch only for smoke tests"
        )

    text_path = Path(train_config.text_cache_path) if train_config.text_cache_path else None
    if model_config.stage == "irr" and text_path is None:
        raise ValueError("Stage 1a requires --text-cache")
    dataset = DepthPairDataset(
        Path(train_config.manifest_path), image_size=model_config.image_size, text_cache_path=text_path
    )
    val_dataset = DepthPairDataset(
        Path(train_config.val_manifest_path), image_size=model_config.image_size, text_cache_path=text_path
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=train_config.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_depth_pairs,
        persistent_workers=train_config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_depth_pairs,
        persistent_workers=train_config.num_workers > 0,
    )
    if not len(loader):
        raise ValueError("training dataset is smaller than one batch")

    model, transfer_report = create_model(model_config, train_config, device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.precision == "fp16")
    start_step = 0
    if train_config.resume_checkpoint:
        resume = torch.load(train_config.resume_checkpoint, map_location="cpu", weights_only=False)
        if resume["stage"] != model_config.stage or resume["model_config"] != model_config.to_dict():
            raise ValueError("resume checkpoint stage/model configuration mismatch")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume.get("scaler", {}))
        start_step = int(resume["step"])
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    base = unwrap(model)
    base.train()

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_metrics.jsonl"
    consumed_microbatches = start_step * train_config.grad_accumulation
    epoch = consumed_microbatches // len(loader)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(consumed_microbatches % len(loader)):
        next(iterator)
    started = time.time()
    dtype = torch.bfloat16 if train_config.precision == "bf16" else torch.float16
    best_loss = float("inf")
    for step in range(start_step + 1, train_config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logged_loss = 0.0
        logged_reconstruction = 0.0
        for accumulation_index in range(train_config.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            depth = batch["depth_pair"].to(device, non_blocking=True)
            language = batch.get("language")
            language_mask = batch.get("language_mask")
            if language is not None:
                language = language.to(device, non_blocking=True)
                language_mask = language_mask.to(device, non_blocking=True)
            sync_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel)
                and accumulation_index < train_config.grad_accumulation - 1
                else nullcontext()
            )
            with sync_context:
                with torch.autocast("cuda", dtype=dtype):
                    outputs = model(depth, language, language_mask)
                    loss = outputs["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
                scaler.scale(loss / train_config.grad_accumulation).backward()
            logged_loss += float(loss.detach())
            logged_reconstruction += float(outputs["reconstruction_loss"].detach())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(base.trainable_parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        restarted = 0
        if train_config.restart_every and step % train_config.restart_every == 0:
            quantizer = base.ti_codebook if model_config.stage == "irr" else base.tc_codebook
            assert quantizer is not None
            restarted = quantizer.restart_dead(optimizer)

        if rank == 0 and (step == 1 or step % train_config.log_every == 0 or step == train_config.steps):
            row = {
                "step": step,
                "loss": logged_loss / train_config.grad_accumulation,
                "reconstruction_loss": logged_reconstruction / train_config.grad_accumulation,
                "grad_norm": float(grad_norm),
                "restarted_codes": restarted,
                "elapsed_seconds": time.time() - started,
                "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                "effective_global_batch": effective_batch,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        if rank == 0 and train_config.save_every and step % train_config.save_every == 0:
            package = checkpoint_package(base, optimizer, scaler, step, model_config, train_config, transfer_report)
            atomic_torch_save(package, output_dir / f"step_{step:07d}.pt")

        if train_config.eval_every and (step % train_config.eval_every == 0 or step == train_config.steps):
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                metrics = validate(base, val_loader, device, max_batches=min(64, len(val_loader)))
                metrics["step"] = step
                with (output_dir / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                active_keys = [key for key in metrics if key.endswith("_active_fraction")]
                entropy_keys = [key for key in metrics if key.endswith("_normalized_entropy")]
                eligible = all(metrics[key] >= 0.75 for key in active_keys) and all(
                    metrics[key] >= 0.5 for key in entropy_keys
                )
                if eligible and metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    package = checkpoint_package(
                        base, optimizer, scaler, step, model_config, train_config, transfer_report
                    )
                    atomic_torch_save(package, output_dir / "best.pt")
            if world_size > 1:
                dist.barrier()

    if rank == 0:
        package = checkpoint_package(
            base, optimizer, scaler, train_config.steps, model_config, train_config, transfer_report
        )
        final_path = output_dir / "last.pt"
        atomic_torch_save(package, final_path)
        metrics = validate(base, val_loader, device, max_batches=min(64, len(val_loader)))
        (output_dir / "validation.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    else:
        final_path = output_dir / "last.pt"
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return final_path


def load_task_checkpoint(path: Path, device: torch.device) -> DepthFactorizedLAM:
    package = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(package["model_config"])
    model = DepthFactorizedLAM(config)
    model.load_state_dict(package["model"], strict=True)
    model.to(device).eval()
    return model
