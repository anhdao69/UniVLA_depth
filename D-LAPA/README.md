# LAPA-Depth


## Getting started

Install the core Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the repository environment:

```bash
export LAPA_ROOT="$(pwd -P)"
export PYTHONPATH="$LAPA_ROOT:${PYTHONPATH:-}"
```

If separate Python environments are used for model inference and LIBERO simulation, set:

```bash
export MODEL_PY=/path/to/model/python
export LIBERO_PY=/path/to/libero/python
```

## Data Preparation

```text
<private-data-root>/
  images/
  <train-split>.jsonl
  <evaluation-split>.jsonl
  <evaluation-bin-edges>.csv
  <depth-feature-shards>/

<private-checkpoint-root>/
  <tokenizer-model>
  <visual-tokenizer-checkpoint>/
  <base-policy-parameters>/
  <stage25-checkpoint>
  <depth-estimator-checkpoint>
```

The training JSONL files should contain one robot demonstration frame per line with at least:

```json
{"instruction": "...", "image": "images/<suite>/<task>/<episode>/<step>.jpg", "raw_actions": [...], "action": ["..."], "fields": "[instruction],[vision],action"}
```
## Stage-3 Training

Launch one suite with explicit command-line configuration:

```bash
bash scripts/train_lapa_depth_suite.sh \
  --model <feature-extractor-name> \
  --suite <training-split-name> \
  --data-root /path/to/prepared/data \
  --feature-root /path/to/offline/depth/features \
  --tokenizer /path/to/tokenizer.model \
  --vqgan /path/to/visual/tokenizer/checkpoint \
  --init-params /path/to/base/policy/parameters \
  --output-dir /path/to/output/root \
  --experiment-id <anonymous-run-id> \
  --total-steps 20000 \
  --batch-size 128 \
  --mesh-dim '!-1,4,1,1' \
  --lr 2e-5 \
  --save-model-freq 20000 \
  --save-milestone-freq 5000 \
  --keep-last-milestones 1 \
  --save-optimizer-state true
```

`streaming_params` is the params-only checkpoint used for rollout. `streaming_train_state`
is the full optimizer state used for exact resume. Milestone files such as
`streaming_train_state_15000` are controlled by `--save-milestone-freq`; with
`--keep-last-milestones 1`, only the newest milestone step is kept.

## Online Rollout Evaluation

Run split online rollout for multiple suites:

```bash
bash scripts/eval_lapa_depth_split_multi_suite.sh \
  --model <feature-extractor-name> \
  --suites "<split-a> <split-b> <split-c>" \
  --checkpoint-template '/path/to/outputs/128_batch_{model}_{suite}/streaming_params' \
  --stage25-checkpoint '/path/to/depth/models/{model}.65000.pt' \
  --action-bin-template '/path/to/data/action_bins_{suite}.csv' \
  --output-root /path/to/evaluation/output/root \
  --task-ids "0 1 2 3 4 5 6 7 8 9" \
  --n-eval-per-task 10 \
  --max-steps 350 \
  --policy-gpus 2 \
  --stage25-gpus 0 \
  --rgb-gpus 1 \
  --egl-gpu 0
```

For a single policy checkpoint shared across suites, use:

```bash
bash scripts/eval_lapa_depth_split_multi_suite.sh \
  --model <feature-extractor-name> \
  --suites "<split-a> <split-b> <split-c>" \
  --shared-checkpoint /path/to/shared/streaming_params \
  --action-fusion concat
```
## License

This code builds on public research software for LAPA-style policy learning, LIBERO simulation, and depth estimation. Follow the licenses and citation requirements of all upstream software and datasets used to run the experiments.
