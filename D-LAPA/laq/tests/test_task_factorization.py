import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from laq.task_factorization.config import ModelConfig
from laq.task_factorization.data import build_pair_manifests, load_jsonl, read_depth
from laq.task_factorization.model import DepthFactorizedLAM, VectorQuantizer


def tiny_config(stage: str) -> ModelConfig:
    return ModelConfig(
        stage=stage,
        image_size=32,
        patch_size=8,
        dim=32,
        encoder_depth=1,
        decoder_depth=1,
        heads=4,
        mlp_ratio=2,
        num_queries=4,
        vq_dim=8,
        irr_codebook_size=5,
        task_codebook_size=3,
        text_dim=12,
    )


def test_uint16_depth_is_preserved(tmp_path: Path):
    values = np.array([[0, 1], [32768, 65535]], dtype=np.uint16)
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), values)
    actual = read_depth(path, image_size=2).squeeze(0).numpy()
    np.testing.assert_allclose(actual, values.astype(np.float32) / 65535.0, rtol=0, atol=1e-7)


def test_manifest_uses_only_exact_delta_and_unique_ids(tmp_path: Path):
    depth_root = tmp_path / "depth"
    annotations = []
    for video_id in range(1, 5):
        directory = depth_root / str(video_id)
        directory.mkdir(parents=True)
        annotations.append(
            {
                "id": str(video_id),
                "label": f"moving object {video_id}",
                "template": "moving [something]",
                "placeholders": [f"object {video_id}"],
            }
        )
        for frame in (1, 2, 3, 4):
            image = np.full((4, 4), frame * 100, dtype=np.uint16)
            assert cv2.imwrite(str(directory / f"img{frame:04d}.png"), image)
    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(json.dumps(annotations))
    output_dir = tmp_path / "manifest"
    result = build_pair_manifests(
        depth_root, annotation_path, output_dir, delta=2, val_fraction=0.25, seed=42, verify_all=True
    )
    rows = load_jsonl(output_dir / "train_pairs.jsonl") + load_jsonl(output_dir / "val_pairs.jsonl")
    assert result["train_pairs"] + result["val_pairs"] == 8
    assert all(row["future_t"] - row["t"] == 2 for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)
    assert all("__tp" in row["id"] for row in rows)


def test_vq_losses_are_fp32_and_indices_valid():
    quantizer = VectorQuantizer(num_codes=4, code_dim=8)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    output = quantizer(x)
    assert output.codebook_loss.dtype == torch.float32
    assert output.commitment_loss.dtype == torch.float32
    assert output.quantized.dtype == torch.bfloat16
    assert output.indices.shape == (2, 4)
    assert int(output.indices.min()) >= 0 and int(output.indices.max()) < 4


def test_two_stage_shapes_transfer_and_frozen_ti_codebook():
    torch.manual_seed(7)
    depth = torch.rand(2, 2, 1, 32, 32)
    language = torch.rand(2, 5, 12)
    language_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)

    stage1 = DepthFactorizedLAM(tiny_config("irr"))
    stage1_output = stage1(depth, language, language_mask)
    assert stage1_output["irr_tokens"].shape == (2, 4, 32)
    assert stage1_output["irr_indices"].shape == (2, 4)
    assert stage1_output["reconstruction"].shape == (2, 1, 32, 32)
    stage1_output["loss"].backward()
    assert stage1.language_proj.weight.grad is not None

    stage2 = DepthFactorizedLAM(tiny_config("task"))
    report = stage2.initialize_task_from_stage1(stage1.state_dict())
    assert report["missing"] == [] and report["incompatible"] == []
    assert not stage2.ti_codebook.codebook.weight.requires_grad
    with pytest.raises(ValueError, match="must not receive language"):
        stage2(depth, language, language_mask)

    before = stage2.ti_codebook.codebook.weight.detach().clone()
    optimizer = torch.optim.AdamW(stage2.trainable_parameters(), lr=1e-3)
    output = stage2(depth)
    assert output["task_feature"].shape == (2, 32)
    assert output["task_tokens"].shape == (2, 4, 32)
    assert output["task_quantized"].shape == (2, 4, 8)
    assert output["task_indices"].shape == (2, 4)
    output["loss"].backward()
    assert stage2.tc_queries.grad is not None
    assert stage2.ti_queries.grad is not None
    optimizer.step()
    torch.testing.assert_close(stage2.ti_codebook.codebook.weight, before, rtol=0, atol=0)

    teacher = stage2.extract_teacher(depth[:, 0], depth[:, 1])
    assert set(teacher) == {
        "task_feature", "task_tokens", "task_indices", "task_quantized",
        "irr_feature", "irr_tokens", "irr_indices", "irr_quantized",
    }
