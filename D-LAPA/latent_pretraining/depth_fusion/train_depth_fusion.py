import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from latent_pretraining.depth_fusion.data_libero import (
    LiberoDepthFusionDataset,
    discover_part_files,
    load_manifest,
)
from latent_pretraining.depth_fusion.model import DepthFusionConfig, DepthFusionPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train LIBERO depth-fusion action head.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rgb_data_dir", type=Path, default=None)
    parser.add_argument("--rgb_manifest", type=Path, default=None)
    parser.add_argument("--action_data_dir", type=Path, default=None)
    parser.add_argument("--action_manifest", type=Path, default=None)
    parser.add_argument("--action_jsonl", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rgb_feature_key", type=str, default="auto")
    parser.add_argument("--depth_feature_key", type=str, default="auto")
    parser.add_argument("--action_key", type=str, default="auto")
    parser.add_argument("--image_key", type=str, default="auto")
    parser.add_argument("--rgb_id_key", type=str, default="auto")
    parser.add_argument("--action_id_key", type=str, default="auto")
    parser.add_argument("--action_jsonl_id_key", type=str, default="id")
    parser.add_argument("--action_jsonl_key", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train, max_batches=None):
    model.train(train)
    total_loss = 0.0
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(tqdm(loader, leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break

            rgb_feature = batch["rgb_feature"].to(device)
            depth_feature = batch["depth_feature"].to(device)
            action = batch["action"].to(device)

            pred = model(rgb_feature, depth_feature)
            loss = criterion(pred, action)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = action.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest if args.manifest is not None else None
    manifest = load_manifest(manifest_path)
    part_files = discover_part_files(args.data_dir, manifest_path)
    rgb_manifest_path = args.rgb_manifest if args.rgb_manifest is not None else None
    rgb_manifest = load_manifest(rgb_manifest_path)
    rgb_part_files = (
        discover_part_files(args.rgb_data_dir, rgb_manifest_path)
        if args.rgb_data_dir is not None
        else None
    )
    action_manifest_path = args.action_manifest if args.action_manifest is not None else None
    action_manifest = load_manifest(action_manifest_path)
    action_part_files = (
        discover_part_files(args.action_data_dir, action_manifest_path)
        if args.action_data_dir is not None
        else None
    )
    dataset = LiberoDepthFusionDataset(
        part_files=part_files,
        manifest=manifest,
        rgb_part_files=rgb_part_files,
        rgb_manifest=rgb_manifest,
        action_part_files=action_part_files,
        action_manifest=action_manifest,
        action_jsonl=args.action_jsonl,
        action_jsonl_id_key=args.action_jsonl_id_key,
        action_jsonl_key=args.action_jsonl_key,
        rgb_feature_key=args.rgb_feature_key,
        depth_feature_key=args.depth_feature_key,
        action_key=args.action_key,
        image_key=args.image_key,
        rgb_id_key=args.rgb_id_key,
        action_id_key=args.action_id_key,
        preload=not args.no_preload,
    )
    print(
        json.dumps(
            {
                "resolved_keys": {
                    "rgb_feature_key": dataset.rgb_feature_key,
                    "depth_feature_key": dataset.depth_feature_key,
                    "action_key": dataset.action_key,
                    "image_key": dataset.image_key,
                },
                "num_parts": len(part_files),
                "num_rgb_parts": len(rgb_part_files) if rgb_part_files is not None else 0,
                "num_action_parts": len(action_part_files) if action_part_files is not None else 0,
                "num_samples": len(dataset),
            },
            indent=2,
        )
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    config = DepthFusionConfig(hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = DepthFusionPolicy(config).to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            args.device,
            train=True,
            max_batches=args.max_train_batches,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            args.device,
            train=False,
            max_batches=args.max_val_batches,
        )
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(json.dumps(record))

        checkpoint = {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, args.output_dir / "best.pt")

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
