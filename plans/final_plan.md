# Task-Centric Depth Encoder via UniVLA Language Factorization

## Summary

Implement a new depth-pixel teacher alongside the existing D-LAPA Stage-1 code. It will preserve the downstream contracts required by the future Model 7 heads—four discrete task codes with `K_task=8` and one continuous 1024-D task feature—while using UniVLA's shared query-VQ language factorization.

The key correction to `plans/train_encoder.md` is the second factorization stage. Faithful UniVLA does **not** freeze a separate language-conditioned encoder and detach its output. Stage 1b removes language completely, introduces task-centric queries and a task codebook, transfers the shared backbone, and freezes only the task-irrelevant codebook embedding. The shared encoder, decoder, TI queries/projections, and new TC branch remain trainable.

The primary experiment stays in normalized depth-pixel space. It does not use masks or DINO features. Stage 2.5 policy training and Stage 3 LIBERO integration are downstream work, but this implementation must produce compatible teacher targets and controls.

## Data and Language Pipeline

### Dataset audit and pair manifest

- Add an `audit-data` command that accepts explicit depth-root and annotation-path arguments. It may discover candidate annotations, but it must record and print the exact resolved file and schema before continuing.
- For the current snapshot, the expected inputs are `/scratch/11528/anhdao69/data/ssv2/depth_train` and `/scratch/11528/anhdao69/data/ssv2/labels/train.json`. Treat previously observed counts—30,000 source videos, 26,171 usable videos, and 470,549 exact pairs—as snapshot expectations, not loader constants.
- Inspect the depth-generation code when available and sample the current PNGs to verify dtype, numeric range, normalization convention, and resolution. Proceed with `float32 / 65535` only after confirming that files are `uint16` on a `[0,65535]` scale.
- Read depth with `cv2.IMREAD_UNCHANGED`. Never use PIL `.convert("RGB")`, which destroys the 16-bit values.
- Resize to `256×256` with nearest-neighbor interpolation and return one channel. Do not perform another per-load min-max normalization.
- Enumerate only exact `(t,t+30)` pairs. Exclude videos with fewer than 31 frames; never clamp the future index.
- Define the canonical unique identifier as `{video_id}__t{t:06d}__tp{t_plus_30:06d}`. Store it as `pair_id` and `id`, while retaining `video_id`, `t`, and `future_t` separately. Fail on duplicate IDs.
- Manifest generation must fail on missing annotations, corrupt frames, inconsistent frame numbering, duplicate IDs, or an unverified depth encoding.
- Create a deterministic seed-42, template-stratified 95/5 split at the video level. No video may occur in both splits.
- Enumerate every valid pair and sample training pairs uniformly. Keep the validation manifest fixed and evaluate it without random temporal sampling.
- Apply no flips, crops, color transforms, or directional augmentation.

### Instruction contract

- Keep `label`, `template`, and `placeholders` as distinct fields.
- Use the verified SSv2 `label` string as the primary instruction.
- Construct and store `filled_instruction` from `template` and `placeholders` only for auditing and the later language-string ablation; do not use it in the primary run.
- Use a locally supplied, revision-pinned `t5-base` model. Run it frozen and in evaluation mode.
- Precompute token-level 768-D T5 hidden states and attention masks, with a maximum sequence length of 64. Fail instead of truncating if an instruction exceeds the limit.
- Store the text-model path/identifier, revision, tokenizer settings, and cache hash in the data manifest and every checkpoint.

## Feature Contract

### Result of the Model-4 audit

The checked-in D-LAPA Stage-1 model does not export an existing 1024-D Model-4 target: training returns reconstruction loss/code usage, inference returns reconstruction or indices, and Model 4 only consumes an externally supplied continuous target. Therefore, the repository cannot reproduce a unique historical tap point from public code alone.

Formalize the following pre-VQ contract and record it as a design clarification:

- **Legacy D-LAPA baseline:** after the spatial and temporal encoders, compute
  `legacy_feature = mean(last_tokens - first_tokens, spatial dimensions)`.
  The input tensors are `[B,1,8,8,1024]`, so the result is `[B,1024]`.
  This tap is before `NSVQ.project_in`, quantization, and decoder conditioning.
- **Factorized teacher:** extract the four future-timestep query states after the shared encoder and before the branch's `1024→32` VQ projection. Define
  `task_feature = mean(task_tokens, dim=1)` and
  `irr_feature = mean(irr_tokens, dim=1)`, both `[B,1024]`.
- Apply no extra learned projection or target normalization. Export the features as float32.

Add a baseline extraction method and a unit test that checks this formula numerically against the existing encoder states. If an original external Model-4 target shard is later located, compare it with this definition and report the discrepancy; do not silently redefine the new target.

## Shared Query-VQ Depth Model

Implement the new code under `D-LAPA/laq/task_factorization/` rather than modifying the device-hardcoded NSVQ and positional-encoding implementations.

### Common architecture

- Input pair: `[B,2,1,256,256]`.
- Patch size: 32, yielding 64 patch tokens per frame.
- Transformer width: 1024, 8 causal spatiotemporal encoder blocks, 8 spatial decoder blocks, 16 heads, and dropout 0.
- Four learned query tokens per latent branch.
- Main VQ embedding width: 32. Keep `vq_dim=128` as a later ablation.
- Quantizer: straight-through nearest-neighbor VQ with a trainable embedding table.
- Compute nearest-neighbor distances, codebook loss, and commitment loss in FP32 outside autocast. The surrounding Transformer may use mixed precision.
- VQ objective: codebook loss plus `0.25 × commitment loss`.
- Project each quantized 32-D code back to 1024-D before decoder conditioning.
- Predict 64 future single-channel depth patches without an output clamp. Clamp only for visualization.

### Exact token flow

For both stages, prepend queries independently at each of the two timesteps, run the causal spatiotemporal encoder, and extract branch states only from the future timestep `t=1`.

Stage 1a encoder:

~~~text
per-timestep visual sequence = [TI query × 4, depth patch × 64]
language memory             = projected T5 tokens with padding mask
TI states                   = encoded future-timestep positions 0:4
~~~

Stage 1b encoder:

~~~text
per-timestep visual sequence = [TC query × 4, TI query × 4, depth patch × 64]
TC states                    = encoded future-timestep positions 0:4
TI states                    = encoded future-timestep positions 4:8
language                     = absent
~~~

Stage 1a decoder sequence is `[q_TI × 4, current-depth patches × 64]`, with direct masked attention access to the full projected T5 token sequence. Stage 1b decoder sequence is `[q_TC × 4, q_TI × 4, current-depth patches × 64]` with no language input. Return and score only the 64 reconstructed depth-patch positions.

### Stage 1a: task-irrelevant pretraining

- Use `K_irr=16` and four TI queries.
- Project frozen T5 states from 768 to 1024.
- Give both encoder and decoder token-level, padding-masked access to the full T5 sequence; do not replace it with a pooled language bias.
- Optimize future-depth MSE, TI codebook loss, and `0.25 ×` TI commitment loss.
- During validation, compare correct instructions with a deterministic shuffled-language control drawn from the validation instruction pool while explicitly excluding each sample's original instruction. Do not rely on a simple in-batch permutation, which can leave duplicated labels unchanged.

### Stage 1b: task-centric factorization

- Create four new TC queries and a new `K_task=8` task codebook.
- Transfer all compatible non-language Stage-1a parameters: depth patch embedding, positional parameters, shared encoder, decoder, TI queries, TI projections, and TI codebook.
- Use an explicit transfer map. Fail on every missing or unexpected key except an allowlist containing the removed language modules and newly initialized TC modules.
- Remove T5, language projections, text caches, and language arguments from the Stage-1b computation graph.
- Freeze only the 16-entry TI codebook embedding and exclude it from the optimizer. Keep all shared components, TI queries/projections, TC queries/projections, and the task codebook trainable.
- Do not detach `q_irr`. Straight-through reconstruction and TI commitment gradients must continue updating its query path and the shared encoder.
- Optimize future-depth MSE, task codebook loss, `0.25 ×` task commitment loss, and `0.25 ×` TI commitment loss. A TI codebook loss may be logged but has no optimizable term because that embedding is frozen.

## Training and Checkpointing

- Stage 1a: 25,000 optimizer steps.
- Stage 1b: 25,000 optimizer steps initialized from the selected Stage-1a checkpoint.
- Global batch size 64, AdamW, learning rate `1e-4`, weight decay `1e-2`, default Adam betas, gradient clipping at `0.1`, and no learning-rate scheduler.
- Use DDP and mixed precision. Use gradient accumulation only to preserve global batch 64, and fail if world size/local batch/accumulation do not multiply to 64.
- Record world size, local batch, accumulation, precision, seed, and effective batch in the run manifest.
- Aggregate VQ assignment counts across ranks over each 1,000-step window.
- A code is dead when it has zero global assignments during that full window. On rank 0, replace dead **trainable** entries from randomly selected globally active entries plus small noise, broadcast the result, and zero the corresponding rows of Adam's `exp_avg` and `exp_avg_sq`. Leave Adam's parameter-level scalar step unchanged and record the restart. If no code is active, reinitialize the complete trainable codebook from its original initialization distribution and zero both moment tensors.
- Never restart or alter the frozen TI codebook in Stage 1b.
- Evaluate reconstruction and code statistics every 1,000 steps. Save resumable `last` and periodic 2,500-step checkpoints.
- Select `best` as the lowest validation total-loss checkpoint satisfying the code-usage/entropy gates. Run language sensitivity, task-code ablations, and linear probes on `best`, `last`, and the three lowest-loss eligible checkpoints; the final exported checkpoint must pass every acceptance gate.
- Checkpoints must contain schema version, stage, model/config state, optimizer/scaler state, step, RNG states, run manifest, data/text-cache hashes, T5 provenance, and the parent Stage-1a checkpoint hash.
- Train the corrected legacy single-branch D-LAPA teacher for 50,000 steps on the same manifest as a step-matched baseline. Repeat the normalized one-channel input to three channels only at the legacy model boundary. Call the comparison compute-matched only if measured GPU-hours or FLOPs are also matched and reported.

## CLI and Public Interfaces

The supported invocation starts from `D-LAPA`, where `laq` is importable:

~~~bash
cd /work2/11528/anhdao69/stampede3/code/UniVLA/D-LAPA

python -m laq.task_factorization.cli audit-data
python -m laq.task_factorization.cli cache-text
python -m laq.task_factorization.cli train-irr
python -m laq.task_factorization.cli train-task --stage1-checkpoint ...
python -m laq.task_factorization.cli evaluate
python -m laq.task_factorization.cli export
~~~

Add a CLI smoke test that runs `--help` from this directory. Do not rely on the repository root implicitly adding `D-LAPA` to `PYTHONPATH`.

The final Stage-1b model exposes an instruction-free API:

~~~python
extract_teacher(depth_t, depth_future) -> {
    "task_feature":   FloatTensor[B, 1024],
    "task_tokens":    FloatTensor[B, 4, 1024],
    "task_indices":   LongTensor[B, 4],       # values 0..7
    "task_quantized": FloatTensor[B, 4, 32],
    "irr_feature":    FloatTensor[B, 1024],
    "irr_tokens":     FloatTensor[B, 4, 1024],
    "irr_indices":    LongTensor[B, 4],       # values 0..15
    "irr_quantized":  FloatTensor[B, 4, 32],
}
~~~

- `task_feature` is the future Model 7b MSE-plus-cosine target.
- `task_indices` is the future Model 7a cross-entropy target.
- TI fields support the task-irrelevant controls. A TI CE head must use 16 classes.
- Extraction is deterministic in evaluation mode and never loads T5.

### Feature-shard schema

Use PyTorch `.pt` parts plus the repository's JSON manifest style, not NPZ. Default to 4,096 examples per part.

Each part contains:

~~~text
id                        list[str], canonical unique pair IDs
pair_id                   list[str], identical to id
video_id                  list[str]
t                         LongTensor[N]
future_t                  LongTensor[N]
depth_t_path              list[str]
depth_future_path         list[str]
z_depth_feature_gt        FloatTensor[N, 1024]   # TC target
z_depth_indices_gt        LongTensor[N, 4]
z_depth_feature_irr       FloatTensor[N, 1024]
z_depth_indices_irr       LongTensor[N, 4]
~~~

The JSON manifest records `id_key="id"`, `feature_key="z_depth_feature_gt"`, `indices_key="z_depth_indices_gt"`, TI control keys, part paths, sample counts, tensor shapes/dtypes, checkpoint hash, and pair-manifest hash.

Validate uniqueness across all parts before finalizing the manifest. Downstream RGB-feature extraction must consume the same pair manifest and preserve `pair_id`; positional alignment without matching IDs is forbidden.

## Validation and Acceptance Gates

Automated tests must cover:

- Audit-gated uint16 decoding and verified `/65535` normalization.
- Exact-Δ sampling, short-video exclusion, pair-ID uniqueness, split isolation, label lookup, and deterministic manifest hashes.
- Numerical verification of the formalized legacy 1024-D pre-VQ feature.
- Exact Stage-1a and Stage-1b token ordering, future-timestep query extraction, output slicing, and language padding masks.
- Sensitivity of both Stage-1a encoder and decoder to language.
- Complete absence of language from the Stage-1b API and graph.
- All public tensor shapes, dtypes, code ranges, loss terms, CPU/GPU portability, and deterministic extraction.
- FP32 VQ distances and losses under surrounding bf16/fp16 autocast.
- Strict and audited Stage-1a→Stage-1b transfer.
- A Stage-1b optimizer step proving the TI embedding remains bitwise unchanged while the shared encoder/decoder and both query paths receive gradients.
- Distributed assignment aggregation, synchronized code restart, optimizer-state reset, and protection of the frozen TI table.
- Checkpoint resume/round-trip equivalence, CLI `--help`, and a finite real-data smoke run.
- `.pt` shard round-trip, manifest key resolution, global pair-ID uniqueness, and feature/RGB ID alignment.

A checkpoint is eligible for downstream use only when:

- Correct Stage-1a instructions produce lower held-out reconstruction MSE than the deterministic different-instruction control, with a paired-bootstrap 95% confidence interval for the improvement excluding zero.
- Full Stage-1b reconstruction is better than reconstruction with independently shuffled task codes and with zeroed task codes. Both improvements must have paired-bootstrap 95% confidence intervals excluding zero.
- At least 75% of each codebook is active on the full validation split, overall normalized entropy is at least 0.5, and no query position is constant.
- Linear probes train only on training-video features and evaluate only on held-out videos. Use the same regularized linear-probe selection protocol for TC and TI features.
- Probe both continuous 1024-D features and position-wise one-hot indices for balanced 174-template classification and ridge prediction of the attention-mask-weighted mean frozen T5 embedding.
- The task branch must outperform the TI branch on held-out balanced template accuracy, with a bootstrap 95% confidence interval for the difference excluding zero. Otherwise, mark factorization as failed and do not start Model 7 training.

## Controlled Experiments

After the primary checkpoint passes:

1. Compare with the corrected 50k-step single-branch D-LAPA baseline.
2. Train Stage 1a without decoder language conditioning.
3. Sweep `K_irr ∈ {8,16,32}`, with 16 as the primary setting.
4. Compare `vq_dim=128` with the primary D-LAPA-compatible `vq_dim=32`.
5. Compare the primary `label` instruction with the stored `filled_instruction`.
6. Train downstream Model 7a/7b from TC targets and matching control heads from TI targets.
7. Run the depth-free Model 7-noD downstream control.
8. Treat DINO or cross-modal reconstruction as later, separately reported experiments.

## Assumptions

- SSv2 is the only teacher-training dataset in this plan.
- LIBERO language continues to reach the downstream policy through the existing language-conditioned RGB stream; it is not an input to Stage 1b or teacher extraction.
- Stored depth is per-frame relative depth, not metric depth.
- Pixel-space reconstruction is the primary experiment.
- `K_task=8`, four task codes, `vq_dim=32`, and the formalized 1024-D mean pre-VQ query feature are fixed primary-run contracts.
- Stage 2.5 and Stage 3 implementation changes remain outside this task.
- The incomplete `plans/train_encoder.md` draft remains audit history; this file is the implementation source of truth.
