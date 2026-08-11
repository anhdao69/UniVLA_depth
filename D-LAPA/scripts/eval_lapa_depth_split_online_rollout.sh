#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

MODEL_PY="${MODEL_PY:-python}"
LIBERO_PY="${LIBERO_PY:-$MODEL_PY}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
LIBERO_REPO="${LIBERO_REPO:?Set LIBERO_REPO to the simulator repository}"
DEPTH_BRANCH_ROOT="${DEPTH_BRANCH_ROOT:?Set DEPTH_BRANCH_ROOT to the Stage-2.5 source bundle}"
SUITE="${SUITE:?Set SUITE to the evaluation split}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:?Set ACTION_SCALE_FILE to this split's bin-edge CSV}"
ACTION_FUSION_METHOD="${ACTION_FUSION_METHOD:-project}"
case "$ACTION_FUSION_METHOD" in
  project|concat) ;;
  *) echo "ERROR: ACTION_FUSION_METHOD must be project or concat" >&2; exit 1 ;;
esac

FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:?Set FINETUNED_CHECKPOINT using the params:: prefix}"
ORIGINAL_LAPA_CHECKPOINT="${ORIGINAL_LAPA_CHECKPOINT:?Set ORIGINAL_LAPA_CHECKPOINT using the params:: prefix}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:?Set VQGAN_CHECKPOINT to the visual tokenizer checkpoint}"
VOCAB_FILE="${VOCAB_FILE:?Set VOCAB_FILE to the tokenizer model}"

STAGE25_MODEL_NAME="${STAGE25_MODEL_NAME:?Set STAGE25_MODEL_NAME to the configured feature extractor name}"
STAGE25_MODEL_CHECKPOINT="${STAGE25_MODEL_CHECKPOINT:?Set STAGE25_MODEL_CHECKPOINT to its checkpoint file}"
DEPTH_ESTIMATOR_REQUIRED="${DEPTH_ESTIMATOR_REQUIRED:-true}"
DEPTH_ANYTHING_REPO_DIR="${DEPTH_ANYTHING_REPO_DIR:-}"
DEPTH_ANYTHING_CHECKPOINT="${DEPTH_ANYTHING_CHECKPOINT:-}"
DEPTH_ANYTHING_ENCODER="${DEPTH_ANYTHING_ENCODER:-vitl}"
DEPTH_ANYTHING_INPUT_SIZE="${DEPTH_ANYTHING_INPUT_SIZE:-518}"
DEPTH_ANYTHING_DEVICE="${DEPTH_ANYTHING_DEVICE:-cuda}"

POLICY_PORT="${POLICY_PORT:-32820}"
STAGE25_PORT="${STAGE25_PORT:-32821}"
RGB_PORT="${RGB_PORT:-32822}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a writable evaluation-output directory}"
LOG_DIR="${LOG_DIR:?Set LOG_DIR to a writable server-log directory}"

POLICY_CUDA_VISIBLE_DEVICES="${POLICY_CUDA_VISIBLE_DEVICES:-1}"
STAGE25_CUDA_VISIBLE_DEVICES="${STAGE25_CUDA_VISIBLE_DEVICES:-0}"
RGB_CUDA_VISIBLE_DEVICES="${RGB_CUDA_VISIBLE_DEVICES:-2}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

POLICY_MESH_DIM="${POLICY_MESH_DIM:-1,1,1,1}"
RGB_MESH_DIM="${RGB_MESH_DIM:-1,1,1,1}"

TASK_IDS="${TASK_IDS:-0}"
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
INIT_OFFSET="${INIT_OFFSET:-0}"
PROGRESS_FREQ="${PROGRESS_FREQ:-25}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

[[ -d "$DEPTH_BRANCH_ROOT" ]] || { echo "ERROR: DEPTH_BRANCH_ROOT not found: $DEPTH_BRANCH_ROOT" >&2; exit 1; }
[[ -f "$STAGE25_MODEL_CHECKPOINT" ]] || { echo "ERROR: STAGE25_MODEL_CHECKPOINT not found: $STAGE25_MODEL_CHECKPOINT" >&2; exit 1; }
[[ -f "$ACTION_SCALE_FILE" ]] || { echo "ERROR: ACTION_SCALE_FILE not found: $ACTION_SCALE_FILE" >&2; exit 1; }
if [[ -z "${ACTION_VOCAB_SIZE:-}" ]]; then
  ACTION_VOCAB_SIZE="$(awk -F, 'NR == 1 { print NF; exit }' "$ACTION_SCALE_FILE")"
fi
if [[ -z "$ACTION_VOCAB_SIZE" || "$ACTION_VOCAB_SIZE" == "0" ]]; then
  echo "ERROR: could not infer ACTION_VOCAB_SIZE from $ACTION_SCALE_FILE" >&2
  exit 1
fi
if [[ -z "${UPDATE_LLAMA_CONFIG:-}" ]]; then
  UPDATE_LLAMA_CONFIG="dict(action_vocab_size=${ACTION_VOCAB_SIZE},depth_feature_dim=1024,action_fusion_method='${ACTION_FUSION_METHOD}',delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)"
fi
if [[ "$DEPTH_ESTIMATOR_REQUIRED" == "true" ]]; then
  [[ -n "$DEPTH_ANYTHING_REPO_DIR" ]] || { echo "ERROR: set DEPTH_ANYTHING_REPO_DIR" >&2; exit 1; }
  [[ -n "$DEPTH_ANYTHING_CHECKPOINT" ]] || { echo "ERROR: set DEPTH_ANYTHING_CHECKPOINT" >&2; exit 1; }
  [[ -d "$DEPTH_ANYTHING_REPO_DIR/depth_anything_v2" ]] || { echo "ERROR: DEPTH_ANYTHING_REPO_DIR is not a DepthAnythingV2 repo: $DEPTH_ANYTHING_REPO_DIR" >&2; exit 1; }
  [[ -f "$DEPTH_ANYTHING_CHECKPOINT" ]] || { echo "ERROR: DEPTH_ANYTHING_CHECKPOINT not found: $DEPTH_ANYTHING_CHECKPOINT" >&2; exit 1; }
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
RGB_LOG="$LOG_DIR/rgb_feature_gpu${RGB_CUDA_VISIBLE_DEVICES//,/}.log"
STAGE25_LOG="$LOG_DIR/stage25_split_gpu${STAGE25_CUDA_VISIBLE_DEVICES//,/}.log"
POLICY_LOG="$LOG_DIR/policy_gpu${POLICY_CUDA_VISIBLE_DEVICES//,/}.log"

cleanup() {
  echo "[split-rollout] stopping servers"
  kill "${POLICY_PID:-}" "${STAGE25_PID:-}" "${RGB_PID:-}" 2>/dev/null || true
  wait "${POLICY_PID:-}" 2>/dev/null || true
  wait "${STAGE25_PID:-}" 2>/dev/null || true
  wait "${RGB_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_log() {
  local name="$1"
  local log="$2"
  local pid="$3"
  echo "[split-rollout] waiting for $name: $log"
  for i in $(seq 1 "${SERVER_WAIT_SECONDS:-360}"); do
    if grep -q "Uvicorn running" "$log"; then
      echo "[split-rollout] READY $name"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[split-rollout] ERROR: $name exited early"
      tail -120 "$log" || true
      return 1
    fi
    if (( i % 30 == 0 )); then
      echo "[split-rollout] still waiting ${i}s for $name"
      tail -40 "$log" || true
      nvidia-smi || true
    fi
    sleep 1
  done
  echo "[split-rollout] ERROR: timed out waiting for $name"
  tail -120 "$log" || true
  return 1
}

echo "[split-rollout] suite: $SUITE"
echo "[split-rollout] action fusion: $ACTION_FUSION_METHOD"
echo "[split-rollout] finetuned policy: $FINETUNED_CHECKPOINT"
echo "[split-rollout] action bins: $ACTION_SCALE_FILE"
echo "[split-rollout] RGB baseline GPU: $RGB_CUDA_VISIBLE_DEVICES mesh=$RGB_MESH_DIM port=$RGB_PORT"
echo "[split-rollout] Stage2.5 GPU: $STAGE25_CUDA_VISIBLE_DEVICES port=$STAGE25_PORT"
echo "[split-rollout] Policy GPU: $POLICY_CUDA_VISIBLE_DEVICES mesh=$POLICY_MESH_DIM port=$POLICY_PORT"
echo "[split-rollout] Simulator EGL GPU: $MUJOCO_EGL_DEVICE_ID"
echo "[split-rollout] Task IDs: $TASK_IDS"
echo "[split-rollout] Eval per task: $N_EVAL_PER_TASK | max_steps=$MAX_STEPS | progress_freq=$PROGRESS_FREQ"

CUDA_VISIBLE_DEVICES="$RGB_CUDA_VISIBLE_DEVICES" "$MODEL_PY" -m eval.lapa_rgb_feature_server \
  --stage25_bundle_dir "$DEPTH_BRANCH_ROOT" \
  --original_lapa_checkpoint "$ORIGINAL_LAPA_CHECKPOINT" \
  --vqgan_checkpoint "$VQGAN_CHECKPOINT" \
  --vocab_file "$VOCAB_FILE" \
  --mesh_dim "$RGB_MESH_DIM" \
  --host "127.0.0.1" \
  --port "$RGB_PORT" \
  > "$RGB_LOG" 2>&1 &
RGB_PID=$!

stage25_args=(
  -m eval.stage25_feature_server
  --stage25_bundle_dir "$DEPTH_BRANCH_ROOT"
  --model_name "$STAGE25_MODEL_NAME"
  --model_checkpoint "$STAGE25_MODEL_CHECKPOINT"
  --original_lapa_checkpoint "$ORIGINAL_LAPA_CHECKPOINT"
  --vqgan_checkpoint "$VQGAN_CHECKPOINT"
  --vocab_file "$VOCAB_FILE"
  --mesh_dim "1,1,1,1"
  --host "127.0.0.1"
  --port "$STAGE25_PORT"
  --rgb_feature_server_url "http://127.0.0.1:${RGB_PORT}"
)
if [[ "$DEPTH_ESTIMATOR_REQUIRED" == "true" ]]; then
  stage25_args+=(
    --depth_anything_repo_dir "$DEPTH_ANYTHING_REPO_DIR"
    --depth_anything_checkpoint "$DEPTH_ANYTHING_CHECKPOINT"
    --depth_anything_encoder "$DEPTH_ANYTHING_ENCODER"
    --depth_anything_input_size "$DEPTH_ANYTHING_INPUT_SIZE"
    --depth_anything_device "$DEPTH_ANYTHING_DEVICE"
  )
fi
CUDA_VISIBLE_DEVICES="$STAGE25_CUDA_VISIBLE_DEVICES" "$MODEL_PY" "${stage25_args[@]}" \
  > "$STAGE25_LOG" 2>&1 &
STAGE25_PID=$!

CUDA_VISIBLE_DEVICES="$POLICY_CUDA_VISIBLE_DEVICES" "$MODEL_PY" -m latent_pretraining.deploy \
  --load_checkpoint "$FINETUNED_CHECKPOINT" \
  --action_scale_file "$ACTION_SCALE_FILE" \
  --vqgan_checkpoint "$VQGAN_CHECKPOINT" \
  --vocab_file "$VOCAB_FILE" \
  --update_llama_config "$UPDATE_LLAMA_CONFIG" \
  --port "$POLICY_PORT" \
  --mesh_dim "$POLICY_MESH_DIM" \
  --tokens_per_delta 4 \
  --tokens_per_action 7 \
  --stage25_feature_server_url "http://127.0.0.1:${STAGE25_PORT}" \
  > "$POLICY_LOG" 2>&1 &
POLICY_PID=$!

wait_for_log "rgb_feature" "$RGB_LOG" "$RGB_PID"
wait_for_log "stage25" "$STAGE25_LOG" "$STAGE25_PID"
wait_for_log "policy" "$POLICY_LOG" "$POLICY_PID"

MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
MUJOCO_EGL_DEVICE_ID="$MUJOCO_EGL_DEVICE_ID" PYTHONPATH="$LIBERO_REPO:$PROJECT_DIR:${PYTHONPATH:-}" \
"$LIBERO_PY" "$PROJECT_DIR/eval/eval_libero_rollout_depth.py" \
  --server_url "http://127.0.0.1:${POLICY_PORT}/act" \
  --output_dir "$OUTPUT_DIR" \
  --suites "$SUITE" \
  --task_ids $TASK_IDS \
  --n_eval_per_task "$N_EVAL_PER_TASK" \
  --max_steps "$MAX_STEPS" \
  --init_offset "$INIT_OFFSET" \
  --progress_freq "$PROGRESS_FREQ"

echo "[split-rollout] wrote results/videos to $OUTPUT_DIR"
