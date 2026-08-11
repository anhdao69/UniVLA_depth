# Task-factorized depth teacher

This package implements the two-stage depth teacher described in
`plans/final_plan.md`. Run commands from the `D-LAPA` directory so the `laq`
package resolves correctly.

## Environment

The validated environment is:

- Python 3.10.20
- PyTorch 2.7.1+cu128
- NVIDIA H100
- Transformers 4.49.0
- SentencePiece 0.2.0

Install the package-specific addition with:

~~~bash
/work2/11528/anhdao69/stampede3/envs/fastwam/bin/pip install -r \
  laq/task_factorization/requirements.txt
~~~

Download and pin T5 outside the training loop:

~~~bash
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 hf download t5-base \
  config.json model.safetensors spiece.model tokenizer.json tokenizer_config.json \
  --local-dir /scratch/11528/anhdao69/models/t5-base
~~~

Set `TMPDIR` to node-local storage in Slurm jobs. TACC's default scratch-backed
temporary directory can produce harmless but noisy `.nfs*` DataLoader cleanup
errors.

## Production sequence

~~~bash
cd /work2/11528/anhdao69/stampede3/code/UniVLA/D-LAPA
export TMPDIR=/tmp
PY=/work2/11528/anhdao69/stampede3/envs/fastwam/bin/python

$PY -m laq.task_factorization.cli audit-data \
  --depth-root /scratch/11528/anhdao69/data/ssv2/depth_train \
  --annotations /scratch/11528/anhdao69/data/ssv2/labels/train.json \
  --output-dir /scratch/11528/anhdao69/data/ssv2/task_factorization_manifest

$PY -m laq.task_factorization.cli cache-text \
  --manifests \
    /scratch/11528/anhdao69/data/ssv2/task_factorization_manifest/train_pairs.jsonl \
    /scratch/11528/anhdao69/data/ssv2/task_factorization_manifest/val_pairs.jsonl \
  --model-path /scratch/11528/anhdao69/models/t5-base \
  --output /scratch/11528/anhdao69/data/ssv2/task_factorization_t5.pt

torchrun --standalone --nproc_per_node=4 -m laq.task_factorization.cli train-irr \
  --manifest /scratch/11528/anhdao69/data/ssv2/task_factorization_manifest/train_pairs.jsonl \
  --val-manifest /scratch/11528/anhdao69/data/ssv2/task_factorization_manifest/val_pairs.jsonl \
  --text-cache /scratch/11528/anhdao69/data/ssv2/task_factorization_t5.pt \
  --output-dir /scratch/11528/anhdao69/checkpoints/depth_factorization/stage1a \
  --steps 25000 --batch-size 16

torchrun --standalone --nproc_per_node=4 -m laq.task_factorization.cli train-task \
  --manifest /scratch/11528/anhdao69/data/ssv2/task_factorization_manifest/train_pairs.jsonl \
  --val-manifest /scratch/11528/anhdao69/data/ssv2/task_factorization_manifest/val_pairs.jsonl \
  --stage1-checkpoint /scratch/11528/anhdao69/checkpoints/depth_factorization/stage1a/best.pt \
  --output-dir /scratch/11528/anhdao69/checkpoints/depth_factorization/stage1b \
  --steps 25000 --batch-size 16
~~~

Use `--resume PATH_TO_LAST_PT` with the same model arguments to continue an
interrupted stage. If local batch 16 does not fit, reduce it and set
`--grad-accumulation` so `world_size × local_batch × accumulation = 64`.
`--allow-nonproduction-batch` exists only for smoke tests.
