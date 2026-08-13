# Task-Centric Depth Encoder: Next Steps and Usage Tutorial

Date: 2026-08-12
Status: Stage 1a and Stage 1b training completed (25,000/25,000 steps each, exit
code 0, no errors). This document covers what to do with the resulting
checkpoints and how to load/run the encoder.

Checkpoints produced:

```text
/scratch/11528/anhdao69/checkpoints/depth_factorization/stage1a/best.pt
/scratch/11528/anhdao69/checkpoints/depth_factorization/stage1b/best.pt
```

Codebook health at the end of training (from `validation_metrics.jsonl`):
TI active fraction 1.0 / entropy ~0.97, task active fraction 1.0 / entropy
~0.91 in both stages. No collapse. This only confirms training ran cleanly —
it does **not** confirm the language factorization worked. That is a separate,
causal question answered in step 1 below.

---

## 1. Run the acceptance-gate diagnostics (required before trusting the checkpoint)

`plans/final_plan.md` treats these as hard gates, not optional sanity checks.
All three need a GPU (the login node has none); run via `idev` or a short
sbatch job the same way training was submitted.

### 1a. Stage 1a language-sensitivity control

Does the decoder actually use the instruction `c`? A correct instruction
should reconstruct better than a deliberately wrong one, with a 95% bootstrap
CI excluding zero.

```bash
cd /work2/11528/anhdao69/stampede3/code/UniVLA/D-LAPA
PY=/work2/11528/anhdao69/stampede3/envs/fastwam/bin/python
MANIFEST=/scratch/11528/anhdao69/data/ssv2/task_factorization_manifest
CKPT=/scratch/11528/anhdao69/checkpoints/depth_factorization

$PY -m laq.task_factorization.cli evaluate \
  --checkpoint $CKPT/stage1a/best.pt \
  --manifest $MANIFEST/val_pairs.jsonl \
  --text-cache /scratch/11528/anhdao69/data/ssv2/task_factorization_t5.pt \
  --output $CKPT/gate_stage1a.json
```

Check `instruction_mse_improvement` and `instruction_mse_improvement_ci95` in
the output: improvement should be positive (wrong instruction reconstructs
worse) and the CI should not straddle zero.

### 1b. Stage 1b task-code ablation control

Does `q_task` carry information the decoder actually needs? Zeroing or
shuffling it should hurt reconstruction.

```bash
$PY -m laq.task_factorization.cli evaluate \
  --checkpoint $CKPT/stage1b/best.pt \
  --manifest $MANIFEST/val_pairs.jsonl \
  --output $CKPT/gate_stage1b.json
```

Check `zero_task_mse_increase_ci95` and `shuffled_task_mse_increase_ci95`:
both should be positive with CIs excluding zero.

### 1c. The linear probe — the actual factorization test (Method 2 §2.4)

```bash
$PY -m laq.task_factorization.cli probe \
  --checkpoint $CKPT/stage1b/best.pt \
  --train-manifest $MANIFEST/train_pairs.jsonl \
  --val-manifest $MANIFEST/val_pairs.jsonl \
  --text-cache /scratch/11528/anhdao69/data/ssv2/task_factorization_t5.pt \
  --output $CKPT/probe_stage1b.json
```

This trains matched linear classifiers from `task_feature`/`task_codes` and
`irr_feature`/`irr_codes` to the 174 SSv2 templates on held-out videos. The
number that matters is `task_minus_irr_feature_accuracy` and its `_ci95`:
**it must be positive with a CI excluding zero.** If `q_irr` predicts the
template as well as or better than `q_task`, the factorization silently
failed and nothing downstream of it is meaningful. This is a hard stop, not
a "proceed with caution."

---

## 2. Only after the gates pass: the controlled-experiment matrix

From `final_plan.md`'s "Controlled Experiments" section, in priority order:

1. **Distill Model 7a/7b from TC targets, and a matching control head from TI
   targets** — the sharp test. If a policy head trained on `q_irr` performs
   at or below a no-depth baseline while one trained on `q_task` beats it,
   that is a clean causal result.
2. Compare against the legacy 50k-step single-branch D-LAPA baseline
   (compute/FLOPs-matched if calling it compute-matched).
3. Stage 1a *without* decoder language conditioning — confirms the §2.3
   "decoder must see `c`" detail is load-bearing (should degrade badly).
4. `K_irr ∈ {8, 16, 32}` sweep.
5. `vq_dim=128` vs the primary `vq_dim=32`.
6. `label` vs `filled_instruction` as the Stage 1a language string.
7. `Model 7-noD` depth-free control.
8. DINO/cross-modal reconstruction — explicitly deferred to later.

---

## 3. Export the feature shards for downstream training

Once satisfied with `stage1b/best.pt`, materialize the teacher targets over
the full manifests — this is what Model 7a/7b actually train against:

```bash
$PY -m laq.task_factorization.cli export \
  --checkpoint $CKPT/stage1b/best.pt \
  --manifest $MANIFEST/train_pairs.jsonl \
  --output-dir /scratch/11528/anhdao69/data/ssv2/task_factorization_features/train

$PY -m laq.task_factorization.cli export \
  --checkpoint $CKPT/stage1b/best.pt \
  --manifest $MANIFEST/val_pairs.jsonl \
  --output-dir /scratch/11528/anhdao69/data/ssv2/task_factorization_features/val
```

Each `.pt` part (4,096 pairs per part) carries `z_depth_feature_gt` (Model
7b's continuous MSE+cos target), `z_depth_indices_gt` (Model 7a's CE target),
and the matching `_irr` fields for the control head — indexed by the same
`pair_id` as the depth manifest, so downstream RGB-feature extraction can
join on it.

---

## 4. Outside this task's scope: Stage 2.5 / Stage 3

`train_encoder.md` deliberately draws the line at producing `E_task`. Not
yet implemented anywhere in the repo:

- **Stage 2.5**: train the causal, single-frame student —
  `(D_t, z^{rgb-feat}_t) → ẑ_t^{task}` — distilled from the exported shards
  via Model 7a (CE against `z_depth_indices_gt`) and/or Model 7b (MSE+cos
  against `z_depth_feature_gt`), plus the `q_irr`-distilled control head
  from the same shards.
- **Stage 3**: fine-tune the LIBERO policy with the causal student's
  task-centric depth feature fused alongside the existing
  language-conditioned RGB stream.
- **LIBERO rollout**: evaluate with the existing D-LAPA harness
  (`Depth_branch/laq/test_ssv2_25_model4_no_gt.py`,
  `rollout_stage25_model4.py`).

---

## 5. Loading and using the encoder directly in Python

For ad-hoc use outside the CLI, the API is `extract_teacher(depth_t,
depth_future)` on a loaded Stage-1b model — it takes single-channel depth
tensors, returns task/TI features and codes, and never touches T5:

```python
import sys
sys.path.insert(0, "/work2/11528/anhdao69/stampede3/code/UniVLA/D-LAPA")

import torch
from laq.task_factorization.trainer import load_task_checkpoint
from laq.task_factorization.data import read_depth

device = torch.device("cuda")
model = load_task_checkpoint(
    "/scratch/11528/anhdao69/checkpoints/depth_factorization/stage1b/best.pt",
    device,
)  # loads config + weights, calls .eval() for you

# read_depth returns [1, H, W] (uint16 -> /65535 -> resized). Batch by stacking.
depth_t = torch.stack([
    read_depth("/scratch/11528/anhdao69/data/ssv2/depth_train/12345/img000100.png", image_size=256),
]).to(device)                      # [B, 1, 256, 256]
depth_future = torch.stack([
    read_depth("/scratch/11528/anhdao69/data/ssv2/depth_train/12345/img000130.png", image_size=256),
]).to(device)                      # [B, 1, 256, 256], t+30 frame

with torch.no_grad():
    teacher = model.extract_teacher(depth_t, depth_future)

teacher["task_feature"]    # FloatTensor[B, 1024]  -> Model 7b target
teacher["task_indices"]    # LongTensor[B, 4], 0..7  -> Model 7a CE target
teacher["task_quantized"]  # FloatTensor[B, 4, 32]
teacher["irr_feature"]     # FloatTensor[B, 1024]  -> TI control target
teacher["irr_indices"]     # LongTensor[B, 4], 0..15
```

Things the model enforces rather than silently allowing:

- Only a **Stage 1b** (`stage="task"`) checkpoint works with
  `extract_teacher` — pointing it at `stage1a/best.pt` raises
  `RuntimeError`.
- Never pass `language`/`language_mask` to a Stage 1b model — it raises
  `ValueError` by construction; this is deliberate, not a bug to work
  around.
- `depth_t`/`depth_future` must each be `[B, 1, 256, 256]` float in `[0, 1]`
  — use `read_depth()` rather than hand-rolling normalization, since it
  enforces the uint16-to-`/65535` convention the model was trained on.

For extracting over many pairs at once, don't hand-roll a loop — use the
`export` CLI command in step 3, which handles batching, `DataLoader`
workers, duplicate-ID checks, and atomic `.pt` shard writing.
