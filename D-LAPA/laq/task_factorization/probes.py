from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import DepthPairDataset, collate_depth_pairs
from .trainer import load_task_checkpoint


@torch.no_grad()
def _extract(
    model,
    dataset: DepthPairDataset,
    template_to_id: Dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> Dict[str, torch.Tensor]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_depth_pairs,
    )
    values: Dict[str, list[torch.Tensor]] = {
        "task_feature": [], "irr_feature": [], "task_codes": [], "irr_codes": [],
        "label": [], "text_target": [],
    }
    count = 0
    for batch in loader:
        depth = batch["depth_pair"].to(device)
        outputs = model.extract_teacher(depth[:, 0], depth[:, 1])
        values["task_feature"].append(outputs["task_feature"].cpu())
        values["irr_feature"].append(outputs["irr_feature"].cpu())
        values["task_codes"].append(
            F.one_hot(outputs["task_indices"].cpu(), model.config.task_codebook_size).flatten(1).float()
        )
        values["irr_codes"].append(
            F.one_hot(outputs["irr_indices"].cpu(), model.config.irr_codebook_size).flatten(1).float()
        )
        values["label"].append(
            torch.tensor([template_to_id[str(row["template"])] for row in batch["row"]], dtype=torch.long)
        )
        assert dataset.text_cache is not None
        text_targets = []
        for row in batch["row"]:
            tokens = dataset.text_cache.get(str(row["label"]))
            text_targets.append(tokens.mean(dim=0))
        values["text_target"].append(torch.stack(text_targets))
        count += depth.shape[0]
        if max_samples is not None and count >= max_samples:
            break
    result = {key: torch.cat(items) for key, items in values.items()}
    if max_samples is not None:
        result = {key: value[:max_samples] for key, value in result.items()}
    return result


def _standardize(train: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (val - mean) / std


def _balanced_accuracy(correct: np.ndarray, labels: np.ndarray, classes: int) -> float:
    scores = [correct[labels == index].mean() for index in range(classes) if np.any(labels == index)]
    return float(np.mean(scores))


def _classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    classes: int,
    device: torch.device,
    epochs: int,
) -> tuple[float, np.ndarray]:
    train_x, val_x = _standardize(train_x.float(), val_x.float())
    layer = nn.Linear(train_x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=3e-2, weight_decay=1e-3)
    counts = torch.bincount(train_y, minlength=classes).float().clamp_min(1)
    weights = (counts.sum() / counts).to(device)
    x, y = train_x.to(device), train_y.to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(layer(x), y, weight=weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = layer(val_x.to(device)).argmax(dim=-1).cpu()
    correct = prediction.eq(val_y).numpy()
    labels = val_y.numpy()
    return _balanced_accuracy(correct, labels, classes), correct


def _text_regression(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    epochs: int,
) -> float:
    train_x, val_x = _standardize(train_x.float(), val_x.float())
    layer = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2, weight_decay=1e-2)
    x, y = train_x.to(device), train_y.to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = layer(x)
        loss = F.mse_loss(prediction, y) + 0.1 * (1.0 - F.cosine_similarity(prediction, y).mean())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = layer(val_x.to(device))
        return float(F.cosine_similarity(prediction, val_y.to(device)).mean())


def _bootstrap_macro_difference(
    task_correct: np.ndarray,
    irr_correct: np.ndarray,
    labels: np.ndarray,
    classes: int,
    draws: int = 2000,
) -> Sequence[float]:
    rng = np.random.default_rng(42)
    differences = []
    class_indices = [np.where(labels == index)[0] for index in range(classes) if np.any(labels == index)]
    for _ in range(draws):
        task_scores = []
        irr_scores = []
        for indices in class_indices:
            sample = rng.choice(indices, size=len(indices), replace=True)
            task_scores.append(task_correct[sample].mean())
            irr_scores.append(irr_correct[sample].mean())
        differences.append(np.mean(task_scores) - np.mean(irr_scores))
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def run_probes(
    checkpoint: Path,
    train_manifest: Path,
    val_manifest: Path,
    text_cache: Path,
    output: Path,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    max_train_samples: int | None = 50_000,
    max_val_samples: int | None = 10_000,
    epochs: int = 100,
) -> Dict[str, Any]:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_task_checkpoint(checkpoint, device)
    train_dataset = DepthPairDataset(
        train_manifest, image_size=model.config.image_size, text_cache_path=text_cache
    )
    val_dataset = DepthPairDataset(val_manifest, image_size=model.config.image_size, text_cache_path=text_cache)
    templates = sorted(
        {str(row["template"]) for row in train_dataset.rows}
        | {str(row["template"]) for row in val_dataset.rows}
    )
    template_to_id = {value: index for index, value in enumerate(templates)}
    train_values = _extract(
        model, train_dataset, template_to_id, device, batch_size, num_workers, max_train_samples
    )
    val_values = _extract(
        model, val_dataset, template_to_id, device, batch_size, num_workers, max_val_samples
    )
    classes = len(templates)
    metrics: Dict[str, Any] = {
        "train_samples": int(train_values["label"].numel()),
        "val_samples": int(val_values["label"].numel()),
        "num_templates": classes,
    }
    correctness = {}
    for branch in ("task_feature", "irr_feature", "task_codes", "irr_codes"):
        score, correct = _classifier(
            train_values[branch], train_values["label"], val_values[branch], val_values["label"],
            classes, device, epochs,
        )
        metrics[f"{branch}_balanced_accuracy"] = score
        correctness[branch] = correct
    for branch in ("task_feature", "irr_feature"):
        metrics[f"{branch}_t5_cosine"] = _text_regression(
            train_values[branch], train_values["text_target"],
            val_values[branch], val_values["text_target"], device, epochs,
        )
    metrics["task_minus_irr_feature_accuracy"] = (
        metrics["task_feature_balanced_accuracy"] - metrics["irr_feature_balanced_accuracy"]
    )
    metrics["task_minus_irr_feature_accuracy_ci95"] = _bootstrap_macro_difference(
        correctness["task_feature"], correctness["irr_feature"], val_values["label"].numpy(), classes
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics
