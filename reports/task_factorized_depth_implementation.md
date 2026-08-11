# Task-Factorized Depth Teacher: Implementation Report

Date: 2026-08-11  
Implementation source of truth: `plans/final_plan.md`

## Outcome

Implemented a new `laq.task_factorization` package without modifying the legacy
D-LAPA teacher. The package covers the primary task-centric encoder workflow:

- exact-offset SSv2 depth auditing and pair manifests;
- frozen token-level T5 caching;
- UniVLA-style Stage 1a task-irrelevant training;
- faithful Stage 1b task-centric factorization;
- strict Stage-1a-to-Stage-1b transfer and TI-codebook freezing;
- FP32 vector quantization under mixed precision;
- single-GPU and DDP training, validation, restart, resume, and checkpoints;
- reconstruction/ablation confidence intervals and TC/TI linear probes;
- deterministic task/TI feature extraction and repository-compatible `.pt` shards.

## Added Files

- `D-LAPA/laq/task_factorization/config.py`: validated model configuration and
  production defaults.
- `data.py`: uint16 depth loader, annotation validation, exact-Δ manifest
  generation, video-level split, dataset, text-cache access, and collation.
- `text_cache.py`: local-only frozen T5 encoding with provenance and hashes.
- `model.py`: patch tokenizer, causal spatiotemporal encoder, spatial decoder,
  standard VQ, two-stage model, strict transfer, teacher API, and formalized
  legacy feature extractor.
- `trainer.py`: mixed-precision DDP loop, gradient accumulation, validation,
  code utilization, dead-code restart, checkpoint selection, and resume.
- `cli.py` / `__main__.py`: `audit-data`, `cache-text`, `train-irr`,
  `train-task`, `evaluate`, `export`, and `probe` commands.
- `exporter.py`: atomic `.pt` parts and JSON manifest writer.
- `probes.py`: matched TC/TI template classifiers, code probes, T5 regression,
  and bootstrap confidence intervals.
- `README.md` and `requirements.txt`: exact environment and production command
  sequence.
- `D-LAPA/laq/tests/test_task_factorization.py`: focused data/model/VQ/transfer
  tests.

## Model Semantics

### Stage 1a

Each timestep contains four TI queries followed by depth-patch tokens. Frozen
T5 tokens are projected to model width and supplied to both encoder and decoder
through masked token-level attention. The future TI query states are projected
to 32-D and quantized against a 16-entry codebook. The decoder reconstructs the
future normalized depth from quantized TI tokens, current-depth tokens, and
language.

The optimized loss is:

```text
future-depth MSE + TI codebook loss + 0.25 * TI commitment loss
```

### Stage 1b

Each timestep contains four newly initialized TC queries, four transferred TI
queries, and depth patches. Language modules and arguments do not exist in the
Stage 1b graph. All compatible non-language weights transfer with an explicit
shape-checked allowlist. Only the 16-entry TI embedding table is frozen and
excluded from the optimizer; the TI projections/queries and shared model remain
trainable. `q_irr` is not detached.

The optimized loss is:

```text
future-depth MSE
+ TC codebook loss
+ 0.25 * TC commitment loss
+ 0.25 * TI commitment loss
```

### Continuous feature contract

The public code had no unique historical 1024-D Model-4 producer. The
implementation therefore formalizes the audited contract:

- legacy feature: spatial mean of the encoded `last_tokens - first_tokens`
  before `NSVQ.project_in`;
- TC/TI feature: mean of the respective four future query states after the
  shared encoder and before the branch's 1024-to-32 VQ projection.

No learned pooling or extra target normalization is applied. Production
outputs are `[B,1024]`; discrete outputs are `[B,4]` with task IDs in `0..7`
and TI IDs in `0..15`.

## Data and Provenance

- Depth is read only with `cv2.IMREAD_UNCHANGED` and must be one-channel
  `uint16`.
- Values are converted once with `/65535`, then resized by nearest neighbor.
- Pairs are present only when both exact frame indices `t` and `t+30` exist.
- Canonical IDs include video and both frame indices, preventing silent shard
  collisions.
- The 95/5 split is deterministic, template-stratified, and video-disjoint.
- Primary language is the annotation's `label`; reconstructed template text is
  retained only for the planned ablation.
- T5 inputs are cached as token sequences, never pooled before conditioning.
- Data, text cache, parent checkpoint, and model configuration hashes are stored
  in checkpoint/export metadata.

## Training Reliability

- VQ distances and both VQ losses run in FP32 outside autocast.
- BF16 is the primary mode; FP16 uses `torch.amp.GradScaler`.
- The trainer enforces effective global batch 64 unless an explicit smoke-test
  override is supplied.
- Gradient accumulation uses DDP `no_sync` on non-final microbatches.
- Resume restores model, optimizer, scaler, and step and resumes at the
  deterministic sampler position.
- DDP assignment counts are globally reduced. Dead trainable codes are replaced
  on rank 0, broadcast to every rank, and have the matching Adam moment rows
  reset on every rank. The frozen Stage 1b TI table is never restarted.
- Validation uses a separate manifest. `best.pt` requires utilization and
  entropy gates in addition to lowest validation loss.
- Checkpoint writes and feature-part writes are atomic.

## Public Output Schema

`extract_teacher(depth_t, depth_future)` returns task/TI continuous features,
four query tokens, four indices, and four 32-D quantized vectors. It rejects
language by construction.

Export uses `.pt` parts with canonical IDs, frame metadata,
`z_depth_feature_gt`, `z_depth_indices_gt`, and corresponding TI controls. The
manifest declares the exact ID/feature/index keys. This was checked against the
existing `ShardFieldIndex` loader.

## Environment Changes

Validated environment:

- `/work2/11528/anhdao69/stampede3/envs/fastwam`
- Python 3.10.20
- PyTorch 2.7.1+cu128
- Transformers 4.49.0
- SentencePiece 0.2.0 (installed during this implementation)
- NVIDIA H100 GPUs

T5-base is pinned locally at
`/scratch/11528/anhdao69/models/t5-base`. Set `TMPDIR=/tmp` in compute jobs;
the TACC default points to NFS scratch and caused worker cleanup warnings.

## Boundaries

The primary encoder pipeline is implemented. The full 25k + 25k optimization
and later controlled experiments are not claimed complete by this report. The
legacy 50k control, K/VQ-width ablations, Model 7a/7b policy training, no-depth
control, and DINO/cross-modal studies begin only after a full primary checkpoint
passes the language, task-dependence, utilization, and probe gates.

The implementation is packaged in the parent repository together with the
relevant D-LAPA source, plans, papers, tests, and validation report. Nested Git
metadata, checkpoints, data caches, and smoke artifacts are excluded. No legacy
tracked source was overwritten.
