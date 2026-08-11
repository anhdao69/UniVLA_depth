from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import DepthPairDataset, build_pair_manifests, collate_depth_pairs
from .exporter import export_features
from .probes import run_probes
from .text_cache import build_text_cache
from .trainer import TrainConfig, load_task_checkpoint, train


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--vq-dim", type=int, default=32)
    parser.add_argument("--irr-codebook-size", type=int, default=16)
    parser.add_argument("--task-codebook-size", type=int, default=8)


def model_config(args: argparse.Namespace, stage: str) -> ModelConfig:
    return ModelConfig(
        stage=stage,
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        vq_dim=args.vq_dim,
        irr_codebook_size=args.irr_codebook_size,
        task_codebook_size=args.task_codebook_size,
    )


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--restart-every", type=int, default=1000)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--allow-nonproduction-batch", action="store_true")
    parser.add_argument("--resume", type=Path)
    add_model_arguments(parser)


def train_config(args: argparse.Namespace, *, text_cache=None, stage1_checkpoint=None) -> TrainConfig:
    return TrainConfig(
        manifest_path=str(args.manifest),
        val_manifest_path=str(args.val_manifest),
        output_dir=str(args.output_dir),
        text_cache_path=str(text_cache) if text_cache else None,
        stage1_checkpoint=str(stage1_checkpoint) if stage1_checkpoint else None,
        resume_checkpoint=str(args.resume) if args.resume else None,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        restart_every=args.restart_every,
        precision=args.precision,
        allow_nonproduction_batch=args.allow_nonproduction_batch,
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(package["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_task_checkpoint(args.checkpoint, device) if cfg.stage == "task" else None
    if model is None:
        from .model import DepthFactorizedLAM
        model = DepthFactorizedLAM(cfg)
        model.load_state_dict(package["model"], strict=True)
        model.to(device).eval()
    dataset = DepthPairDataset(
        args.manifest,
        image_size=cfg.image_size,
        text_cache_path=args.text_cache if cfg.stage == "irr" else None,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_depth_pairs,
    )
    sums = {"loss": 0.0, "reconstruction_mse": 0.0}
    paired_differences: dict[str, list[float]] = {}
    alternate_texts = None
    if cfg.stage == "irr":
        assert dataset.text_cache is not None
        available = sorted(dataset.text_cache.entries)
        if len(available) < 2:
            raise ValueError("different-instruction evaluation requires at least two unique instructions")
        alternate_texts = {
            text: available[(index + 1) % len(available)] for index, text in enumerate(available)
        }
        sums["different_instruction_mse"] = 0.0
        paired_differences["instruction_mse_improvement"] = []
    if cfg.stage == "task":
        sums.update({"zero_task_mse": 0.0, "shuffled_task_mse": 0.0})
        paired_differences["zero_task_mse_increase"] = []
        paired_differences["shuffled_task_mse_increase"] = []
    count = 0
    hist_irr = torch.zeros(cfg.irr_codebook_size, dtype=torch.long)
    hist_task = torch.zeros(cfg.task_codebook_size, dtype=torch.long)
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        depth = batch["depth_pair"].to(device)
        language = batch.get("language")
        language_mask = batch.get("language_mask")
        if language is not None:
            language = language.to(device)
            language_mask = language_mask.to(device)
        outputs = model(depth, language, language_mask)
        sums["loss"] += float(outputs["loss"])
        sums["reconstruction_mse"] += float(outputs["reconstruction_loss"])
        hist_irr += torch.bincount(outputs["irr_indices"].cpu().reshape(-1), minlength=cfg.irr_codebook_size)
        full_per_sample = F.mse_loss(outputs["reconstruction"].float(), depth[:, 1].float(), reduction="none")
        full_per_sample = full_per_sample.flatten(1).mean(dim=1)
        if cfg.stage == "irr":
            assert alternate_texts is not None and dataset.text_cache is not None
            controls = [dataset.text_cache.get(alternate_texts[str(row["label"])]) for row in batch["row"]]
            max_length = max(value.shape[0] for value in controls)
            control_language = torch.zeros(len(controls), max_length, cfg.text_dim, device=device)
            control_mask = torch.zeros(len(controls), max_length, dtype=torch.bool, device=device)
            for index, value in enumerate(controls):
                length = value.shape[0]
                control_language[index, :length] = value.to(device)
                control_mask[index, :length] = True
            different = model(depth, control_language, control_mask)
            different_per_sample = F.mse_loss(
                different["reconstruction"].float(), depth[:, 1].float(), reduction="none"
            ).flatten(1).mean(dim=1)
            sums["different_instruction_mse"] += float(different_per_sample.mean())
            paired_differences["instruction_mse_improvement"].extend(
                (different_per_sample - full_per_sample).cpu().tolist()
            )
        if cfg.stage == "task":
            zero = model(depth, task_ablation="zero")
            shuffled = model(depth, task_ablation="shuffle")
            zero_per_sample = F.mse_loss(
                zero["reconstruction"].float(), depth[:, 1].float(), reduction="none"
            ).flatten(1).mean(dim=1)
            shuffled_per_sample = F.mse_loss(
                shuffled["reconstruction"].float(), depth[:, 1].float(), reduction="none"
            ).flatten(1).mean(dim=1)
            sums["zero_task_mse"] += float(zero_per_sample.mean())
            sums["shuffled_task_mse"] += float(shuffled_per_sample.mean())
            paired_differences["zero_task_mse_increase"].extend(
                (zero_per_sample - full_per_sample).cpu().tolist()
            )
            paired_differences["shuffled_task_mse_increase"].extend(
                (shuffled_per_sample - full_per_sample).cpu().tolist()
            )
            hist_task += torch.bincount(outputs["task_indices"].cpu().reshape(-1), minlength=cfg.task_codebook_size)
        count += 1
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    metrics["batches"] = count
    metrics["irr_histogram"] = hist_irr.tolist()
    if cfg.stage == "task":
        metrics["task_histogram"] = hist_task.tolist()
    rng = np.random.default_rng(42)
    for name, values in paired_differences.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size:
            draws = rng.choice(array, size=(2000, array.size), replace=True).mean(axis=1)
            metrics[name] = float(array.mean())
            metrics[f"{name}_ci95"] = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-centric depth factorization")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-data", help="audit depth data and build exact-delta manifests")
    audit.add_argument("--depth-root", type=Path, required=True)
    audit.add_argument("--annotations", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--delta", type=int, default=30)
    audit.add_argument("--val-fraction", type=float, default=0.05)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--max-videos", type=int)
    audit.add_argument("--verify-all", action="store_true")

    cache = sub.add_parser("cache-text", help="cache frozen token-level T5 embeddings")
    cache.add_argument("--manifests", type=Path, nargs="+", required=True)
    cache.add_argument("--model-path", required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--batch-size", type=int, default=64)
    cache.add_argument("--max-length", type=int, default=64)
    cache.add_argument("--device", default="cuda")

    irr = sub.add_parser("train-irr", help="train language-conditioned TI Stage 1a")
    add_train_arguments(irr)
    irr.add_argument("--text-cache", type=Path, required=True)

    task = sub.add_parser("train-task", help="train language-free TC Stage 1b")
    add_train_arguments(task)
    task.add_argument("--stage1-checkpoint", type=Path, required=True)

    eval_parser = sub.add_parser("evaluate", help="evaluate reconstruction and branch dependence")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--manifest", type=Path, required=True)
    eval_parser.add_argument("--text-cache", type=Path)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--num-workers", type=int, default=4)
    eval_parser.add_argument("--max-batches", type=int)

    export = sub.add_parser("export", help="export TC/TI feature shards")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--batch-size", type=int, default=32)
    export.add_argument("--num-workers", type=int, default=8)
    export.add_argument("--part-size", type=int, default=4096)
    export.add_argument("--max-batches", type=int)

    probe = sub.add_parser("probe", help="run matched TC/TI linear probes")
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--train-manifest", type=Path, required=True)
    probe.add_argument("--val-manifest", type=Path, required=True)
    probe.add_argument("--text-cache", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--batch-size", type=int, default=32)
    probe.add_argument("--num-workers", type=int, default=4)
    probe.add_argument("--max-train-samples", type=int, default=50_000)
    probe.add_argument("--max-val-samples", type=int, default=10_000)
    probe.add_argument("--epochs", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit-data":
        result = build_pair_manifests(
            args.depth_root, args.annotations, args.output_dir,
            delta=args.delta, val_fraction=args.val_fraction, seed=args.seed,
            max_videos=args.max_videos, verify_all=args.verify_all,
        )
    elif args.command == "cache-text":
        result = build_text_cache(
            args.manifests, args.model_path, args.output,
            batch_size=args.batch_size, max_length=args.max_length, device=args.device,
        )
    elif args.command == "train-irr":
        result = {"checkpoint": str(train(model_config(args, "irr"), train_config(args, text_cache=args.text_cache)))}
    elif args.command == "train-task":
        result = {
            "checkpoint": str(
                train(model_config(args, "task"), train_config(args, stage1_checkpoint=args.stage1_checkpoint))
            )
        }
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "export":
        result = {
            "manifest": str(
                export_features(
                    args.checkpoint, args.manifest, args.output_dir,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    part_size=args.part_size, max_batches=args.max_batches,
                )
            )
        }
    elif args.command == "probe":
        result = run_probes(
            args.checkpoint, args.train_manifest, args.val_manifest, args.text_cache, args.output,
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples,
            epochs=args.epochs,
        )
    else:
        raise AssertionError(args.command)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
