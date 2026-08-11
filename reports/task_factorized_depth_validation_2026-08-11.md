# Task-Factorized Depth Teacher: Validation Report

Date: 2026-08-11  
Slurm allocation: `3393290`  
Node: `c561-010.stampede3.tacc.utexas.edu`  
Hardware: 4 × NVIDIA H100 (95,830 MiB each)

## Automated Checks

Command:

```bash
cd /work2/11528/anhdao69/stampede3/code/UniVLA/D-LAPA
/work2/11528/anhdao69/stampede3/envs/fastwam/bin/python \
  -m pytest -q laq/tests/test_task_factorization.py
```

Result: **4 passed**.

Covered behavior:

- exact uint16 pixel preservation and `/65535` normalization;
- exact temporal offsets and globally unique pair IDs;
- FP32 VQ loss/distance behavior with BF16 inputs;
- Stage 1a shapes and language gradients;
- strict Stage-1a-to-Stage-1b transfer;
- rejection of language by Stage 1b;
- TC/TI public shapes and index bounds;
- TC and TI query gradients;
- bitwise frozen TI embedding after an optimizer step;
- deterministic teacher extraction keys.

`compileall` and CLI `--help` also passed. Gradient accumulation was exercised
with two microbatches and produced a finite optimizer step/checkpoint.

## Real-Data Audit and T5 Cache

Bounded smoke manifest:

- 64 source videos;
- 61 usable exact-Δ videos;
- 1,098 training pairs;
- 172 validation pairs;
- 3,187 frames observed;
- 128 boundary frames decoded and verified as uint16;
- manifest hash:
  `3999f14166630d88710246ae6474ffb3d9e9d2b8465ef27322a7949b5148ae49`.

The same 64-video subset was then rerun with `--verify-all`. All 3,187 PNGs
decoded as one-channel uint16, the pair counts were identical, and the exhaustive
audit completed with manifest hash
`e468fa652809611440c894a76c0b94e681a8c11d5f970fe6f6506e977ddaecf1`.

The production audit also completed over the full dataset:

- 30,000 source videos and 1,353,947 frames;
- 26,171 usable videos;
- 446,839 training pairs from 24,863 videos;
- 23,710 validation pairs from 1,308 disjoint videos;
- 470,549 total exact `(t,t+30)` pairs;
- 60,000 boundary frames decoded as uint16;
- manifest hash:
  `c5e9ab9e092ae4b8b40ea02376b8ab096a992130b84887ca674d7a299b8f784c`.

T5 cache:

- local model: `/scratch/11528/anhdao69/models/t5-base`;
- 61 unique instructions;
- maximum observed token length: 19 (configured cap: 64);
- cache hash:
  `6dbf6a6b265c46d89b89dd085dfd89b7751bbe41b8977de8124bda36c64752a2`.

The full production T5 cache was then generated successfully:

- 21,313 unique filled labels across both splits;
- maximum token length 30, below the no-truncation cap of 64;
- cache hash:
  `e70b8e5c7860458d8951d884c88cad31f8fc20f34b5e8b3b89e6d1478efd9dfd`;
- output: `/scratch/11528/anhdao69/data/ssv2/task_factorization_t5.pt`.

The first cache attempt correctly failed because SentencePiece was absent. After
installing SentencePiece 0.2.0, local-only T5 loading and caching succeeded.

## GPU Training Runs

All runs used real SSv2 uint16 depth pairs and cached real T5-base tokens.

### Tiny end-to-end debug run

Configuration: 64² input, width 128, 2 encoder + 2 decoder blocks, batch 2,
three steps per stage.

Stage 1a:

| Step | Total loss | Reconstruction MSE | Gradient norm |
|---:|---:|---:|---:|
| 1 | 0.79748 | 0.46394 | 1.0790 |
| 2 | 0.79187 | 0.48652 | 0.9114 |
| 3 | 0.80016 | 0.54096 | 1.2585 |

Stage 1b:

| Step | Total loss | Reconstruction MSE | Gradient norm |
|---:|---:|---:|---:|
| 1 | 1.03840 | 0.52963 | 1.1837 |
| 2 | 0.83073 | 0.38255 | 1.3166 |
| 3 | 0.74928 | 0.35926 | 1.2935 |

The checkpoint resume test continued Stage 1a from step 1 to step 2 with model,
optimizer, scaler, and sampler position restored.

### Exact production architecture, single GPU

Configuration: 256² input, width 1024, 8 encoder + 8 decoder blocks, 16 heads,
VQ width 32.

| Stage | Steps | Loss | Reconstruction MSE | Gradient norm | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| 1a | 1 | 0.96796 | 0.53108 | 3.1627 | 4.50 GiB |
| 1b | 1 | 2.07737 | 0.68399 | 7.5060 | 4.49 GiB |

Both stages completed forward, backward, clipping, optimizer update, validation,
and checkpoint serialization.

### Exact production architecture, four-GPU DDP

Configuration: four H100s, local batch 16, global batch 64, BF16, two optimizer
steps per stage.

Stage 1a:

| Step | Total loss | Reconstruction MSE | Gradient norm | Peak/rank |
|---:|---:|---:|---:|---:|
| 1 | 1.06623 | 0.61400 | 2.8341 | 5.42 GiB |
| 2 | 4.32653 | 0.40333 | 5.0873 | 5.73 GiB |

Stage 1b:

| Step | Total loss | Reconstruction MSE | Gradient norm | Peak/rank |
|---:|---:|---:|---:|---:|
| 1 | 0.93774 | 0.29250 | 3.2854 | 5.39 GiB |
| 2 | 1.75708 | 0.20668 | 3.7850 | 5.51 GiB |

The revised DDP run exited cleanly with one rank-0 result, explicit NCCL device
selection, and process-group destruction.

The same two-stage four-GPU test was repeated using the **full production pair
manifests and full 21,313-instruction T5 cache**. Stage 1a losses were 1.01842
and 4.03505 with reconstruction MSE decreasing from 0.59458 to 0.42933. Stage
1b losses were 0.96780 and 1.79642 with reconstruction MSE decreasing from
0.30364 to 0.21586. Both jobs exited cleanly at global batch 64, and the Stage
1b TI table again remained bitwise equal to its full-manifest Stage 1a parent.

Validation after two steps:

- Stage 1a TI active fraction 0.75; normalized entropy 0.5641.
- Stage 1b task active fraction 1.0; normalized entropy 0.3963.
- Stage 1b TI active fraction 0.4375; normalized entropy 0.1497.

These are software smoke statistics, not converged representation results.

## Projected Full-Training Runtime

The full-manifest four-H100 smoke run measured approximately 0.31 seconds per
steady-state Stage-1a optimizer step and 0.17 seconds per Stage-1b step after
warm-up. At 25,000 optimizer steps per stage, the update-only projections are
about 2.2 hours and 1.2 hours, respectively.

Production runs also perform validation every 1,000 steps, write a checkpoint
every 2,500 steps, and serialize roughly 2.7 GiB checkpoints. Allowing for this
validation, filesystem I/O, startup, final evaluation, probes, and teacher
export, budget **5--6 hours on one four-H100 node** for the primary 25k + 25k
teacher workflow. A single five-hour interactive allocation is therefore tight;
use a six-hour-or-longer job if available, or submit the two stages as separate
jobs. This estimate does not include the downstream Model-7 policy training or
the full ablation matrix, whose runtimes require separate benchmarks.

## Transfer, Freeze, Export, and Loader Checks

- Strict transfer reported no missing or incompatible keys.
- The Stage 1b TI codebook was bitwise equal to its Stage 1a parent after GPU
  optimization.
- Production teacher export produced task/TI features of shape `[1,1024]` and
  indices of shape `[1,4]`.
- The repository's existing `ShardFieldIndex` discovered the exported `.pt`
  part through its JSON manifest and loaded the 1024-D feature successfully.

## Diagnostic Behavior

Evaluation emits paired bootstrap intervals for correct versus different
instructions and full versus shuffled/zero task codes. Linear-probe smoke also
completed for TC/TI continuous features, discrete codes, and T5 regression.

The two-step checkpoints are intentionally **not accepted scientifically**:

- Stage 1a correct-language improvement was negative on four debug examples.
- Stage 1b shuffled-task increase was approximately zero.
- The tiny probe had insufficient samples/training and returned zero balanced
  template accuracy.

This is expected for random initialization plus two or three updates. The gates
are functioning and prevent accidental downstream use of smoke checkpoints.

## Artifact Locations

- All smoke artifacts:
  `/scratch/11528/anhdao69/tmp/univla_depth_factorization_smoke_3393290`
- Production-shape DDP Stage 1a:
  `stage1_ddp_production/last.pt` (about 2.7 GiB)
- Production-shape DDP Stage 1b:
  `stage2_ddp_production/last.pt` (about 2.7 GiB)
- Local T5-base weights:
  `/scratch/11528/anhdao69/models/t5-base/model.safetensors` (about 851 MiB)
- Full pair manifests:
  `/scratch/11528/anhdao69/data/ssv2/task_factorization_manifest`
- Full T5 token cache:
  `/scratch/11528/anhdao69/data/ssv2/task_factorization_t5.pt`
- Full-manifest production-shape smoke checkpoints:
  `/scratch/11528/anhdao69/checkpoints/depth_factorization/smoke_job3393290_stage1a/last.pt`
  and `smoke_job3393290_stage1b/last.pt`

Smoke checkpoints must not be used for Model 7 training. Run the documented
25k + 25k sequence and require every acceptance gate before export.
