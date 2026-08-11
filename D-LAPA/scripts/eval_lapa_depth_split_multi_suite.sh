#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_lapa_depth_split_multi_suite.sh [options]

Options:
  --lapa-root PATH              Repository root. Default: script parent directory.
  --model NAME                  Stage-2.5 model name, e.g. model2/model4/model5. Default: model5.
  --suites "LIST"              Space-separated suites. Default: "libero_spatial libero_object libero_goal".
  --task-ids "LIST"            Space-separated task IDs. Default: "0 1 2 3 4 5 6 7 8 9".
  --n-eval-per-task N          Episodes per task. Default: 10.
  --max-steps N                Max rollout steps. Default: 350.
  --progress-freq N            Progress print frequency. Default: 25.
  --checkpoint-template PATH   Per-suite policy params template. Supports {model}, {suite}, {suffix}.
                                Default: <root>/outputs/128_batch_{model}_{suite}/streaming_params.
  --shared-checkpoint PATH     One policy params path for all suites. May include params:: prefix.
  --action-bin-template PATH   Action-bin CSV template. Supports {model}, {suite}, {suffix}.
                                Default: lapa_libero_v1/action_bins_{suite}.csv if present,
                                else lapa_libero_v2/action_bins.csv.
  --stage25-checkpoint PATH    Stage-2.5 checkpoint template. Supports {model}. Default:
                                <root>/lapa_checkpoints/depth_model/{model}.65000.pt.
  --data-root PATH             Dataset root. Default: lapa_libero_v1 if present, else lapa_libero_v2.
  --output-root PATH           Rollout output root. Default: <root>/rollouts.
  --output-prefix NAME         Output prefix. Default: eval_split_{model}.
  --action-fusion METHOD       project or concat. Default: project.
  --depth-estimator BOOL       true/false. Default: false for model5, true otherwise.
  --export-params BOOL         If true and streaming_params is missing, export from sibling
                                streaming_train_state before rollout. Default: false.
  --export-mesh DIM            Mesh for params export. Default: 1,1,1,1.
  --policy-gpus LIST           CUDA_VISIBLE_DEVICES for policy server. Default: 2.
  --stage25-gpus LIST          CUDA_VISIBLE_DEVICES for Stage-2.5 server. Default: 0.
  --rgb-gpus LIST              CUDA_VISIBLE_DEVICES for RGB feature server. Default: 1.
  --egl-gpu ID                 MUJOCO_EGL_DEVICE_ID. Default: 0.
  --policy-mesh DIM            Policy mesh_dim. Default: 1,1,1,1.
  --rgb-mesh DIM               RGB feature mesh_dim. Default: 1,1,1,1.
  --help                       Show this help.

This script intentionally ignores stale suite-specific exports from the parent
shell. Pass rollout configuration through these options.
EOF
}

replace_placeholders() {
  local template="$1"
  local suite="$2"
  local model="$3"
  local suffix="${suite#libero_}"
  template="${template//\{suite\}/$suite}"
  template="${template//\{suffix\}/$suffix}"
  template="${template//\{model\}/$model}"
  printf '%s' "$template"
}

normalize_checkpoint() {
  local checkpoint="$1"
  checkpoint="params::${checkpoint#params::}"
  printf '%s' "$checkpoint"
}

resolve_data_root() {
  local root="$1"
  if [[ -d "$root/datasets/lapa_libero_v1" ]]; then
    printf '%s' "$root/datasets/lapa_libero_v1"
  elif [[ -d "$root/datasets/lapa_libero_v2" ]]; then
    printf '%s' "$root/datasets/lapa_libero_v2"
  else
    printf '%s' "$root/datasets/lapa_libero_v1"
  fi
}

default_action_template() {
  local data_root="$1"
  if [[ -f "$data_root/action_bins_libero_spatial.csv" ]]; then
    printf '%s' "$data_root/action_bins_{suite}.csv"
  else
    printf '%s' "$data_root/action_bins.csv"
  fi
}

LAPA_ROOT="$PROJECT_DIR"
STAGE25_MODEL_NAME="model5"
SUITES="libero_spatial libero_object libero_goal"
TASK_IDS="0 1 2 3 4 5 6 7 8 9"
N_EVAL_PER_TASK="10"
MAX_STEPS="350"
PROGRESS_FREQ="25"
CHECKPOINT_TEMPLATE=""
SHARED_CHECKPOINT=""
ACTION_BIN_TEMPLATE=""
STAGE25_CHECKPOINT_TEMPLATE=""
DATA_ROOT=""
OUTPUT_ROOT=""
OUTPUT_PREFIX=""
ACTION_FUSION_METHOD="project"
DEPTH_ESTIMATOR_REQUIRED=""
EXPORT_PARAMS="false"
EXPORT_MESH="1,1,1,1"
POLICY_GPUS="2"
STAGE25_GPUS="0"
RGB_GPUS="1"
EGL_GPU="0"
POLICY_MESH="1,1,1,1"
RGB_MESH="1,1,1,1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lapa-root) LAPA_ROOT="$2"; shift 2 ;;
    --model) STAGE25_MODEL_NAME="$2"; shift 2 ;;
    --suites) SUITES="$2"; shift 2 ;;
    --task-ids) TASK_IDS="$2"; shift 2 ;;
    --n-eval-per-task) N_EVAL_PER_TASK="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --progress-freq) PROGRESS_FREQ="$2"; shift 2 ;;
    --checkpoint-template) CHECKPOINT_TEMPLATE="$2"; shift 2 ;;
    --shared-checkpoint) SHARED_CHECKPOINT="$2"; shift 2 ;;
    --action-bin-template) ACTION_BIN_TEMPLATE="$2"; shift 2 ;;
    --stage25-checkpoint) STAGE25_CHECKPOINT_TEMPLATE="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
    --action-fusion) ACTION_FUSION_METHOD="$2"; shift 2 ;;
    --depth-estimator) DEPTH_ESTIMATOR_REQUIRED="$2"; shift 2 ;;
    --export-params) EXPORT_PARAMS="$2"; shift 2 ;;
    --export-mesh) EXPORT_MESH="$2"; shift 2 ;;
    --policy-gpus) POLICY_GPUS="$2"; shift 2 ;;
    --stage25-gpus) STAGE25_GPUS="$2"; shift 2 ;;
    --rgb-gpus) RGB_GPUS="$2"; shift 2 ;;
    --egl-gpu) EGL_GPU="$2"; shift 2 ;;
    --policy-mesh) POLICY_MESH="$2"; shift 2 ;;
    --rgb-mesh) RGB_MESH="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$(resolve_data_root "$LAPA_ROOT")"
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$LAPA_ROOT/rollouts"
fi
if [[ -z "$OUTPUT_PREFIX" ]]; then
  OUTPUT_PREFIX="eval_split_${STAGE25_MODEL_NAME}"
fi
if [[ -z "$CHECKPOINT_TEMPLATE" ]]; then
  CHECKPOINT_TEMPLATE="$LAPA_ROOT/outputs/128_batch_{model}_{suite}/streaming_params"
fi
if [[ -z "$ACTION_BIN_TEMPLATE" ]]; then
  ACTION_BIN_TEMPLATE="$(default_action_template "$DATA_ROOT")"
fi
if [[ -z "$STAGE25_CHECKPOINT_TEMPLATE" ]]; then
  STAGE25_CHECKPOINT_TEMPLATE="$LAPA_ROOT/lapa_checkpoints/depth_model/{model}.65000.pt"
fi
if [[ -z "$DEPTH_ESTIMATOR_REQUIRED" ]]; then
  if [[ "$STAGE25_MODEL_NAME" == "model5" ]]; then
    DEPTH_ESTIMATOR_REQUIRED="false"
  else
    DEPTH_ESTIMATOR_REQUIRED="true"
  fi
fi

case "$ACTION_FUSION_METHOD" in
  project|concat) ;;
  *) echo "ERROR: --action-fusion must be project or concat" >&2; exit 1 ;;
esac
case "$DEPTH_ESTIMATOR_REQUIRED" in
  true|false) ;;
  *) echo "ERROR: --depth-estimator must be true or false" >&2; exit 1 ;;
esac
case "$EXPORT_PARAMS" in
  true|false) ;;
  *) echo "ERROR: --export-params must be true or false" >&2; exit 1 ;;
esac
if [[ -n "$SHARED_CHECKPOINT" && -n "$CHECKPOINT_TEMPLATE" && "$CHECKPOINT_TEMPLATE" != "$LAPA_ROOT/outputs/128_batch_{model}_{suite}/streaming_params" ]]; then
  echo "ERROR: use either --shared-checkpoint or --checkpoint-template, not both" >&2
  exit 1
fi

echo "[multi-suite] root: $LAPA_ROOT"
echo "[multi-suite] model: $STAGE25_MODEL_NAME"
echo "[multi-suite] suites: $SUITES"
echo "[multi-suite] task ids: $TASK_IDS"
echo "[multi-suite] n eval per task: $N_EVAL_PER_TASK"
echo "[multi-suite] max steps: $MAX_STEPS"
echo "[multi-suite] policy template: ${SHARED_CHECKPOINT:-$CHECKPOINT_TEMPLATE}"
echo "[multi-suite] stage25 template: $STAGE25_CHECKPOINT_TEMPLATE"
echo "[multi-suite] action bins: $ACTION_BIN_TEMPLATE"
echo "[multi-suite] output root: $OUTPUT_ROOT"
echo "[multi-suite] action fusion: $ACTION_FUSION_METHOD"
echo "[multi-suite] export params: $EXPORT_PARAMS"

for suite in $SUITES; do
  if [[ -n "$SHARED_CHECKPOINT" ]]; then
    checkpoint="$(normalize_checkpoint "$SHARED_CHECKPOINT")"
  else
    checkpoint="$(normalize_checkpoint "$(replace_placeholders "$CHECKPOINT_TEMPLATE" "$suite" "$STAGE25_MODEL_NAME")")"
  fi
  ckpt_path="${checkpoint#params::}"
  stage25_checkpoint="$(replace_placeholders "$STAGE25_CHECKPOINT_TEMPLATE" "$suite" "$STAGE25_MODEL_NAME")"
  action_scale_file="$(replace_placeholders "$ACTION_BIN_TEMPLATE" "$suite" "$STAGE25_MODEL_NAME")"
  suite_output="$OUTPUT_ROOT/${OUTPUT_PREFIX}_${suite}_tasks$(echo "$TASK_IDS" | wc -w)_eps${N_EVAL_PER_TASK}_max${MAX_STEPS}"
  suite_log_dir="$suite_output/server_logs"

  [[ -f "$action_scale_file" ]] || { echo "[multi-suite] ERROR: action bins not found: $action_scale_file" >&2; exit 1; }
  [[ -f "$stage25_checkpoint" ]] || { echo "[multi-suite] ERROR: Stage-2.5 checkpoint not found: $stage25_checkpoint" >&2; exit 1; }

  if [[ ! -e "$ckpt_path" && "$EXPORT_PARAMS" == "true" ]]; then
    exp_dir="$(dirname "$ckpt_path")"
    [[ -e "$exp_dir/streaming_train_state" ]] || {
      echo "[multi-suite] ERROR: params missing and train-state not found: $exp_dir/streaming_train_state" >&2
      exit 1
    }
    echo "[multi-suite] exporting params from train-state: $exp_dir/streaming_train_state"
    CUDA_VISIBLE_DEVICES="$POLICY_GPUS" bash "$SCRIPT_DIR/train_lapa_depth_suite.sh" \
      --lapa-root "$LAPA_ROOT" \
      --suite "$suite" \
      --model "$STAGE25_MODEL_NAME" \
      --data-root "$DATA_ROOT" \
      --output-dir "$(dirname "$exp_dir")" \
      --experiment-id "$(basename "$exp_dir")" \
      --total-steps 0 \
      --batch-size 1 \
      --mesh-dim "$EXPORT_MESH" \
      --action-fusion "$ACTION_FUSION_METHOD" \
      --action-bins "$action_scale_file" \
      --save-model-freq 1 \
      --save-milestone-freq 0 \
      --save-optimizer-state false \
      --autoresume true
  fi
  [[ -e "$ckpt_path" ]] || { echo "[multi-suite] ERROR: policy params not found: $ckpt_path" >&2; exit 1; }

  echo "============================================================"
  echo "[multi-suite] suite: $suite"
  echo "[multi-suite] policy: $checkpoint"
  echo "[multi-suite] stage25: $stage25_checkpoint"
  echo "[multi-suite] action bins: $action_scale_file"
  echo "[multi-suite] output: $suite_output"
  echo "============================================================"

  pkill -u "$USER" -f "latent_pretraining.deploy" || true
  pkill -u "$USER" -f "eval.stage25_feature_server" || true
  pkill -u "$USER" -f "eval.lapa_rgb_feature_server" || true
  sleep "${SERVER_CLEANUP_SLEEP:-5}"
  rm -rf "$suite_log_dir"
  mkdir -p "$suite_log_dir" "$suite_output"

  env -i \
    HOME="$HOME" \
    USER="${USER:-}" \
    PATH="$PATH" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$LAPA_ROOT:${PYTHONPATH:-}" \
    MODEL_PY="${MODEL_PY:-python}" \
    LIBERO_PY="${LIBERO_PY:-${MODEL_PY:-python}}" \
    LAPA_ROOT="$LAPA_ROOT" \
    LIBERO_REPO="${LIBERO_REPO:-$LAPA_ROOT/datasets/LIBERO}" \
    DEPTH_BRANCH_ROOT="${DEPTH_BRANCH_ROOT:-$LAPA_ROOT/Depth_branch}" \
    ORIGINAL_LAPA_CHECKPOINT="${ORIGINAL_LAPA_CHECKPOINT:-params::$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/params}" \
    VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}" \
    VOCAB_FILE="${VOCAB_FILE:-$(if [[ -f "$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/tokenizer.model" ]]; then printf '%s' "$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/tokenizer.model"; else printf '%s' "$LAPA_ROOT/lapa_checkpoints/tokenizer.model"; fi)}" \
    STAGE25_MODEL_NAME="$STAGE25_MODEL_NAME" \
    STAGE25_MODEL_CHECKPOINT="$stage25_checkpoint" \
    DEPTH_ESTIMATOR_REQUIRED="$DEPTH_ESTIMATOR_REQUIRED" \
    DEPTH_ANYTHING_REPO_DIR="${DEPTH_ANYTHING_REPO_DIR:-$LAPA_ROOT/third_party/depth_anything_v2}" \
    DEPTH_ANYTHING_CHECKPOINT="${DEPTH_ANYTHING_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/depth_model/depth_anything_v2_sth2sth.pth}" \
    DEPTH_ANYTHING_ENCODER="${DEPTH_ANYTHING_ENCODER:-vitl}" \
    DEPTH_ANYTHING_INPUT_SIZE="${DEPTH_ANYTHING_INPUT_SIZE:-518}" \
    DEPTH_ANYTHING_DEVICE="${DEPTH_ANYTHING_DEVICE:-cuda}" \
    POLICY_CUDA_VISIBLE_DEVICES="$POLICY_GPUS" \
    STAGE25_CUDA_VISIBLE_DEVICES="$STAGE25_GPUS" \
    RGB_CUDA_VISIBLE_DEVICES="$RGB_GPUS" \
    MUJOCO_EGL_DEVICE_ID="$EGL_GPU" \
    POLICY_MESH_DIM="$POLICY_MESH" \
    RGB_MESH_DIM="$RGB_MESH" \
    XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}" \
    TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}" \
    JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}" \
    SUITE="$suite" \
    FINETUNED_CHECKPOINT="$checkpoint" \
    ACTION_SCALE_FILE="$action_scale_file" \
    ACTION_VOCAB_SIZE="" \
    ACTION_FUSION_METHOD="$ACTION_FUSION_METHOD" \
    TASK_IDS="$TASK_IDS" \
    N_EVAL_PER_TASK="$N_EVAL_PER_TASK" \
    MAX_STEPS="$MAX_STEPS" \
    PROGRESS_FREQ="$PROGRESS_FREQ" \
    OUTPUT_DIR="$suite_output" \
    LOG_DIR="$suite_log_dir" \
    bash "$SCRIPT_DIR/eval_lapa_depth_split_online_rollout.sh"
done

echo "[multi-suite] all suites complete"
