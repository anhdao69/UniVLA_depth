from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from transformers import T5EncoderModel, T5Tokenizer

from .data import load_jsonl, stable_json_hash


@torch.no_grad()
def build_text_cache(
    manifest_paths: Sequence[Path],
    model_path: str,
    output_path: Path,
    *,
    batch_size: int = 64,
    max_length: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    if not manifest_paths:
        raise ValueError("at least one pair manifest is required")
    rows = [row for path in manifest_paths for row in load_jsonl(path)]
    texts = sorted({str(row["label"]) for row in rows})
    tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True, legacy=True)
    model = T5EncoderModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    model.requires_grad_(False)
    entries: Dict[str, torch.Tensor] = {}
    max_observed = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        lengths = tokenized.attention_mask.sum(dim=1)
        if int(lengths.max()) > max_length:
            offending = batch[int(lengths.argmax())]
            raise ValueError(f"instruction exceeds max_length={max_length}: {offending!r}")
        max_observed = max(max_observed, int(lengths.max()))
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        hidden = model(**tokenized).last_hidden_state.detach().cpu()
        masks = tokenized["attention_mask"].cpu()
        for index, text in enumerate(batch):
            length = int(masks[index].sum())
            entries[text] = hidden[index, :length].to(torch.float16).contiguous()

    config_path = Path(model_path) / "config.json"
    model_sha = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None
    metadata = {
        "schema_version": 1,
        "model_path": str(Path(model_path).resolve()),
        "model_config_sha256": model_sha,
        "max_length": max_length,
        "max_observed_length": max_observed,
        "tokenizer_class": "T5Tokenizer",
        "tokenizer_legacy": True,
        "padding": "longest-per-cache-batch",
        "truncation": False,
        "text_dim": 768,
        "num_instructions": len(entries),
        "source_manifests": [str(path.resolve()) for path in manifest_paths],
        "source_manifest_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in manifest_paths
        },
    }
    metadata["cache_hash"] = stable_json_hash(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save({"metadata": metadata, "entries": entries}, tmp)
    tmp.replace(output_path)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
