# Technical Implementation Plan — Task-Centric Depth Encoder for D-LAPA Using UniVLA-Style Language Factorization

## 0. Mission

Your task is to implement and train a **task-centric depth latent encoder** for D-LAPA using the language-factorization mechanism inspired by the latent-action version of UniVLA.

The training data is located at:

```bash
/scratch/11528/anhdao69/data/ssv2/depth_train
```

The project repository contains a `papers/` directory. Before changing any code, read all relevant papers and existing implementation carefully, especially:

```text
papers/
├── LAPA...
├── UniVLA...      # Learning to Act Anywhere with Task-centric Latent Actions
└── D-LAPA...
```

Do not treat this as a clean-room implementation. The new implementation must reuse the current D-LAPA depth teacher architecture, data conventions, preprocessing, checkpoint conventions, and Stage-2.5 feature interface wherever possible.

The final output of this task is **not yet the online single-frame Stage-2.5 model**.

The output of this task is an offline teacher encoder:

$$
E_{\text{task}}(D_t, D_{t+\Delta}) \rightarrow z_t^{\text{task}}
$$

where $z_t^{\text{task}}$ is the **task-centric continuous depth feature** that will later supervise the causal Stage-2.5 student.

The implementation must also retain the associated VQ representation:

$$
q_t^{\text{task}} = VQ_{\text{task}}(z_t^{\text{task}})
$$

because we eventually want both:

```text
continuous task-centric feature → Model 7b / Model 4-style distillation
discrete task-centric indices   → Model 7a / Model 2-style distillation
```

The preferred continuous output is **1024-D**, matching the existing D-LAPA Stage-2.5 Model 4 target/interface. However, do not invent a new arbitrary `Linear(..., 1024)` projection merely to satisfy this number. First trace exactly where the current D-LAPA teacher obtains the 1024-D `z_depth_feature`, then expose the analogous representation from the new task-centric branch.

---

# 1. Scientific objective

The current D-LAPA depth teacher learns a generic depth transition:

$$
(D_t, D_{t+\Delta}) \rightarrow z_t^D
$$

The reconstruction objective can force this latent to describe every predictable depth change:

$$
z_t^D \approx \big[\text{task motion},\ \text{camera motion},\ \text{background geometry changes},\ \text{other agents},\ \text{irrelevant object motion}\big]
$$

We want instead:

$$
z_t^{\text{task}} \approx \text{task-relevant geometric dynamics}.
$$

Use language only during an intermediate task-irrelevant training stage to factor the transition.

The intended decomposition is:

$$
\Delta D \approx \Delta D_{\text{irr}} + \Delta D_{\text{task}}.
$$

The method therefore requires **two teacher-training stages**.

---

# 2. Core mechanism

## Stage 1a — learn the task-irrelevant branch

Given $D_t$, $D_{t+\Delta}$, and instruction $\ell$, encode the instruction with a frozen text encoder:

$$
c = T5(\ell).
$$

Train:

$$
z_{\text{irr}} = E_{\text{irr}}(D_t, D_{t+\Delta}, c)
$$

$$
q_{\text{irr}} = VQ_{\text{irr}}(z_{\text{irr}})
$$

and reconstruct:

$$
\hat D_{t+\Delta} = Dec(D_t, q_{\text{irr}}, c).
$$

Objective:

$$
\mathcal{L}_{1a} = \mathcal{L}_{\text{recon}} + \lambda_{vq}\, \mathcal{L}_{VQ,\text{irr}}.
$$

The **decoder must receive language**.

This is load-bearing.

If $Dec(D_t, q_{\text{irr}})$ does not receive $c$, then all information needed to predict the task transition must still travel through $q_{\text{irr}}$, and there is no reason for the latent to become task-irrelevant.

The intended intuition is:

```text
language c
    ↓
already tells decoder the semantic task

q_irr
    ↓
uses its limited VQ capacity to explain the remaining visual/depth transition
```

Therefore $q_{\text{irr}}$ is encouraged to absorb nuisance/residual dynamics.

---

## Stage 1b — learn the task-centric branch

After Stage 1a:

* freeze `E_irr`;
* freeze `VQ_irr`;
* disable/freeze the decoder's direct language-conditioning path;
* detach the `q_irr` tensor;
* add a new task encoder;
* add a new task VQ codebook.

The new encoder must **not receive language**:

$$
z_{\text{task}} = E_{\text{task}}(D_t, D_{t+\Delta}).
$$

Quantize:

$$
q_{\text{task}} = VQ_{\text{task}}(z_{\text{task}}).
$$

The frozen nuisance latent is still computed from the Stage-1a branch:

$$
q_{\text{irr}} = \operatorname{stopgrad}\Big[VQ_{\text{irr}}\big(E_{\text{irr}}(D_t, D_{t+\Delta}, c)\big)\Big].
$$

The Stage-1b decoder must receive:

$$
\hat D_{t+\Delta} = Dec(D_t, q_{\text{irr}}, q_{\text{task}}).
$$

It must **not receive `c` directly**.

Loss:

$$
\mathcal{L}_{1b} = \mathcal{L}_{\text{recon}} + \lambda_{vq}\, \mathcal{L}_{VQ,\text{task}}.
$$

The intuition is:

```text
Stage 1a:
D_t + q_irr + language
    → D_future

Stage 1b:
D_t + frozen q_irr + new q_task
    → D_future
```

The information previously delivered through language must now be represented through the newly introduced task branch.

Thus $q_{\text{task}}$, and especially its continuous precursor $z_{\text{task}}$, are the representations we want.

---

# 3. Critical distinction: language use in Stage 1b

Do not misunderstand "remove language in Stage 1b."

The new **task encoder** must have no language input:

```python
z_task = E_task(depth_t, depth_future)
```

and the Stage-1b decoder must have no direct language input:

```python
depth_future_hat = decoder(
    depth_t,
    q_irr,
    q_task,
)
```

However, the frozen Stage-1a nuisance encoder was trained as:

```python
q_irr = E_irr(depth_t, depth_future, text_condition)
```

so during Stage 1b it is valid and necessary to compute:

```python
with torch.no_grad():
    q_irr = frozen_E_irr(
        depth_t,
        depth_future,
        text_condition,
    )
```

Do **not** silently change the pretrained nuisance encoder's input distribution by removing the text condition from `E_irr`.

The constraint is:

```text
language may be used inside the frozen nuisance teacher
language must NOT directly reach E_task
language must NOT directly reach the Stage-1b reconstruction decoder
```

---

# 4. Scope boundaries

For this task, implement:

```text
dataset audit
SSv2 language-depth alignment
Stage-1a task-irrelevant depth model
Stage-1b task-centric depth model
training scripts
checkpointing
feature extraction
factorization diagnostics
unit/smoke tests
```

Do NOT yet implement:

```text
Stage-2.5 causal student training
LIBERO Stage-3 policy fine-tuning
LIBERO rollout
new policy architecture
new LAPA backbone
mask-based Method 1
```

You may inspect those components only to guarantee interface compatibility.

The endpoint of this task is:

```text
depth pair
   ↓
trained E_task
   ↓
task-centric depth feature
```

---

# 5. Required repository audit before coding

Do not edit code immediately.

First map the existing implementation.

Inspect at minimum:

```text
papers/

laq/
laq/laq_model/
laq/laq_model/latent_action_quantization.py
laq/laq_model/data.py
laq/laq_model/laq_trainer.py
laq/laq_model/nsvq.py
laq/laq_model/t5.py
laq/train_sthv2.py

Depth_branch/
Depth_branch/laq/
Depth_branch/laq/laq_model/
Depth_branch/laq/laq_model/latent_action_quantization_stage25_feature_model4.py
Depth_branch/laq/test_ssv2_25_model4_no_gt.py
Depth_branch/laq/rollout_stage25_model4.py

Depth_branch/README.md

latent_pretraining/
latent_pretraining/depth_fusion/

any current Stage-1 depth-teacher scripts/configs/checkpoints
any code that produces z_depth_feature
any code that produces z_depth_idx
```

Search globally for:

```bash
grep -R "z_depth_feature" -n .
grep -R "z_depth" -n Depth_branch laq
grep -R "codebook" -n Depth_branch laq
grep -R "quant" -n Depth_branch laq
grep -R "1024" -n Depth_branch
grep -R "T5" -n laq Depth_branch
grep -R "depth_scale" -n .
grep -R "65535" -n .
```

Determine and document:

1. the current Stage-1 depth teacher architecture;
2. current depth preprocessing;
3. exact temporal pairing logic;
4. exact tensor representing current `z_depth_feature`;
5. exact tensor representing current VQ indices;
6. whether the model uses NSVQ/VQ-VAE/another quantizer;
7. exact code-length generation mechanism;
8. exact reconstruction decoder input;
9. existing checkpoint structure;
10. how Model 4 consumes the continuous feature.

Write this audit before coding into:

```text
plans/task_centric_depth_encoder_audit.md
```

Do not continue until the 1024-D feature definition is understood.

---

# 6. Read the papers with specific questions

Do not merely summarize the papers.

Extract implementation-relevant answers.

## LAPA

Understand:

```text
How are paired frames encoded?
Where are learnable action queries inserted?
What representation is quantized?
How is code length determined?
How is the future observation reconstructed?
What anti-collapse mechanism is used?
What is stopped-gradient?
```

The new depth implementation should preserve useful structural conventions already inherited from LAPA.

## UniVLA — latent-action paper

Focus only on the task-centric latent-action section.

Understand:

```text
Stage-1 TI representation
language conditioning path
encoder input
decoder input
VQ placement
Stage-2 TC queries
task-centric codebook
which components are initialized from Stage 1
which component is explicitly frozen
why TC is supposed to replace language
```

Do not confuse this paper with the other UniVLA paper based on unified text/image/action tokenization.

## D-LAPA

Understand:

```text
existing depth teacher
existing continuous depth feature
existing discrete depth tokens
Stage-2.5 Model 2
Stage-2.5 Model 4
expected feature width
normalization
current training dataset
how depth feature enters Stage 3
```

Then write a short architecture correspondence:

```text
LAPA generic latent
        ↓
D-LAPA generic depth latent
        ↓
new task-irrelevant depth latent
        ↓
new task-centric depth latent
```

---

# 7. Dataset audit

Dataset:

```bash
/scratch/11528/anhdao69/data/ssv2/depth_train
```

Before training, inspect the complete layout.

Run commands such as:

```bash
find /scratch/11528/anhdao69/data/ssv2/depth_train \
  -maxdepth 2 -type f | head -100

du -sh /scratch/11528/anhdao69/data/ssv2/depth_train

find /scratch/11528/anhdao69/data/ssv2/depth_train \
  -type f | sed 's/.*\.//' | sort | uniq -c
```

Programmatically inspect at least 100 random examples.

Report:

```text
file format
dtype
shape
min
max
mean
percentiles
NaN count
Inf count
normalization
video ID
frame index
number of frames per video
```

Do not assume PNG means uint16 metric depth.

Do not automatically divide by `65535`.

Determine from actual values and existing preprocessing whether stored values are:

```text
uint8 visualization
uint16 depth
float relative depth
normalized [0,1]
inverse-depth/disparity-like
DepthAnything output
```

Use the **exact preprocessing already proven by the D-LAPA baseline**, unless a bug is demonstrated.

---

# 8. Verify SSv2 video-to-language alignment

Method 2 cannot work without language in Stage 1a.

The depth folder alone may not contain the instruction.

Locate the Something-Something-V2 annotations.

Possible locations may be in the repository, adjacent dataset directories, or standard SSv2 annotation files.

Do not invent labels.

For every training video ID, create:

```python
{
    "video_id": ...,
    "instruction": ...,
    "template": ...,
    "placeholders": ...
}
```

Use the **filled instruction**, not only the generic template, when possible.

For example:

```text
template:
"Pushing [something] from left to right"

placeholder:
"a red mug"

filled instruction:
"Pushing a red mug from left to right"
```

If there are multiple placeholders, replace them deterministically in order.

Create an alignment audit reporting:

```text
total depth videos
videos with instruction
videos missing instruction
duplicate IDs
malformed annotations
match rate
```

Acceptance:

```text
match rate should be effectively 100%
```

for the training subset being used.

If not, fail loudly and document missing IDs.

Do not silently use an empty string.

---