#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_lapa_depth_suite.sh [options]

Options:
  --lapa-root PATH              Repository root. Default: script parent directory.
  --suite NAME                  Training split, e.g. libero_object/libero_goal/all. Default: libero_spatial.
  --model NAME                  Stage-2.5 model name, e.g. model2/model4/model5. Default: model5.
  --data-root PATH              LAPA LIBERO JSONL root. Default: <root>/datasets/lapa_libero_v2 if present, else lapa_libero_v1.
  --feature-root PATH           Offline feature root. Default: <root>/datasets/features_depth_branch.
  --train-jsonl PATH            Training JSONL. Default: <data-root>/<suite>.jsonl, or <suite>_train.jsonl if present.
  --image-root PATH             Image root used by JSONL image paths. Default: <data-root>/.
  --depth-dir PATH              Offline depth-feature directory. Supports comma-separated dirs for concat/all-suite.
  --depth-manifest PATH         Optional depth manifest path. Supports comma-separated manifests.
                                If omitted, each depth dir is searched for exactly one *manifest.json.
  --action-bins PATH            Action-bin CSV. Default: <data-root>/action_bins_<suite>.csv, else <data-root>/action_bins.csv.
  --tokenizer PATH              Tokenizer model. Default: <root>/lapa_checkpoints/lapa_7b_sth/tokenizer.model, else tokenizer.model.
  --vqgan PATH                  VQGAN checkpoint. Default: <root>/lapa_checkpoints/vqgan.
  --init-params PATH            Initial LAPA params. Default: <root>/lapa_checkpoints/lapa_7b_sth/params.
  --output-dir PATH             Output root. Default: <root>/outputs.
  --experiment-id NAME          Run directory. Default: 128_batch_<model>_<suite>.
  --total-steps N              Total optimization steps. Default: 20000.
  --batch-size N               Global batch size. Default: 128.
  --seq-length N               Sequence length. Default: 384.
  --mesh-dim DIM               JAX mesh_dim. Default: !-1,4,1,1.
  --lr VALUE                   Learning rate. Default: 2e-5.
  --action-fusion METHOD       project or concat. Default: project.
  --save-model-freq N          Save latest params/train-state frequency. Default: total steps.
  --save-milestone-freq N      Save milestone train-state/params frequency. Default: 0.
  --keep-last-milestones N     Keep only last N milestone steps; 0 keeps all. Default: 1.
  --save-optimizer-state BOOL  Save optimizer state for exact resume. Default: true.
  --autoresume BOOL            Resume from <output>/<experiment>/streaming_train_state. Default: false.
  --wandb-online BOOL          W&B online mode. Default: false.
  --help                       Show this help.

The script uses local variables and launches training with env -i, so stale
suite-specific exports in the parent shell do not override these options.
EOF
}

resolve_data_root() {
  local root="$1"
  if [[ -d "$root/datasets/lapa_libero_v2" ]]; then
    printf '%s' "$root/datasets/lapa_libero_v2"
  elif [[ -d "$root/datasets/lapa_libero_v1" ]]; then
    printf '%s' "$root/datasets/lapa_libero_v1"
  else
    printf '%s' "$root/datasets/lapa_libero_v2"
  fi
}

resolve_train_jsonl() {
  local data_root="$1"
  local suite="$2"
  if [[ -f "$data_root/${suite}_train.jsonl" ]]; then
    printf '%s' "$data_root/${suite}_train.jsonl"
  else
    printf '%s' "$data_root/${suite}.jsonl"
  fi
}

resolve_action_bins() {
  local data_root="$1"
  local suite="$2"
  if [[ -f "$data_root/action_bins_${suite}.csv" ]]; then
    printf '%s' "$data_root/action_bins_${suite}.csv"
  else
    printf '%s' "$data_root/action_bins.csv"
  fi
}

resolve_tokenizer() {
  local root="$1"
  if [[ -f "$root/lapa_checkpoints/lapa_7b_sth/tokenizer.model" ]]; then
    printf '%s' "$root/lapa_checkpoints/lapa_7b_sth/tokenizer.model"
  else
    printf '%s' "$root/lapa_checkpoints/tokenizer.model"
  fi
}

resolve_depth_manifests() {
  local depth_dirs_csv="$1"
  local result=""
  IFS=',' read -ra dirs <<< "$depth_dirs_csv"
  for dir in "${dirs[@]}"; do
    dir="${dir#"${dir%%[![:space:]]*}"}"
    dir="${dir%"${dir##*[![:space:]]}"}"
    [[ -n "$dir" ]] || continue
    [[ -d "$dir" ]] || { echo "ERROR: depth dir not found: $dir" >&2; exit 1; }
    mapfile -t manifests < <(find "$dir" -maxdepth 1 -type f -name '*manifest.json' | sort)
    if [[ "${#manifests[@]}" -ne 1 ]]; then
      echo "ERROR: expected exactly one *manifest.json in $dir, found ${#manifests[@]}" >&2
      printf '  %s\n' "${manifests[@]}" >&2
      exit 1
    fi
    if [[ -z "$result" ]]; then
      result="${manifests[0]}"
    else
      result="$result,${manifests[0]}"
    fi
  done
  printf '%s' "$result"
}

LAPA_ROOT="$PROJECT_DIR"
SUITE="libero_spatial"
STAGE25_MODEL_NAME="model5"
DATA_ROOT=""
FEATURE_ROOT=""
TRAIN_JSONL=""
IMAGE_ROOT=""
DEPTH_DATA_DIR=""
DEPTH_MANIFEST=""
ACTION_SCALE_FILE=""
TOKENIZER_PATH=""
VQGAN_CKPT=""
LAPA_PARAMS=""
OUTPUT_DIR=""
EXPERIMENT_ID=""
TOTAL_STEPS="20000"
BATCH_SIZE="128"
SEQ_LENGTH="384"
MESH_DIM="!-1,4,1,1"
LR="2e-5"
ACTION_FUSION_METHOD="project"
SAVE_MODEL_FREQ=""
SAVE_MILESTONE_FREQ="0"
KEEP_LAST_MILESTONES="1"
SAVE_OPTIMIZER_STATE="true"
AUTORESUME="false"
WANDB_ONLINE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lapa-root) LAPA_ROOT="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --model) STAGE25_MODEL_NAME="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    --train-jsonl) TRAIN_JSONL="$2"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    --depth-dir) DEPTH_DATA_DIR="$2"; shift 2 ;;
    --depth-manifest) DEPTH_MANIFEST="$2"; shift 2 ;;
    --action-bins) ACTION_SCALE_FILE="$2"; shift 2 ;;
    --tokenizer) TOKENIZER_PATH="$2"; shift 2 ;;
    --vqgan) VQGAN_CKPT="$2"; shift 2 ;;
    --init-params) LAPA_PARAMS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seq-length) SEQ_LENGTH="$2"; shift 2 ;;
    --mesh-dim) MESH_DIM="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --action-fusion) ACTION_FUSION_METHOD="$2"; shift 2 ;;
    --save-model-freq) SAVE_MODEL_FREQ="$2"; shift 2 ;;
    --save-milestone-freq) SAVE_MILESTONE_FREQ="$2"; shift 2 ;;
    --keep-last-milestones) KEEP_LAST_MILESTONES="$2"; shift 2 ;;
    --save-optimizer-state) SAVE_OPTIMIZER_STATE="$2"; shift 2 ;;
    --autoresume) AUTORESUME="$2"; shift 2 ;;
    --wandb-online) WANDB_ONLINE="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

DATA_ROOT="${DATA_ROOT:-$(resolve_data_root "$LAPA_ROOT")}"
FEATURE_ROOT="${FEATURE_ROOT:-$LAPA_ROOT/datasets/features_depth_branch}"
TRAIN_JSONL="${TRAIN_JSONL:-$(resolve_train_jsonl "$DATA_ROOT" "$SUITE")}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_ROOT/}"
DEPTH_DATA_DIR="${DEPTH_DATA_DIR:-$FEATURE_ROOT/stage25_libero_features_${STAGE25_MODEL_NAME}/$SUITE/stage25_${STAGE25_MODEL_NAME}/z_depth_train_shard0}"
DEPTH_MANIFEST="${DEPTH_MANIFEST:-$(resolve_depth_manifests "$DEPTH_DATA_DIR")}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$(resolve_action_bins "$DATA_ROOT" "$SUITE")}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$(resolve_tokenizer "$LAPA_ROOT")}"
VQGAN_CKPT="${VQGAN_CKPT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
LAPA_PARAMS="${LAPA_PARAMS:-$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/params}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs}"
EXPERIMENT_ID="${EXPERIMENT_ID:-128_batch_${STAGE25_MODEL_NAME}_${SUITE}}"
SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-$TOTAL_STEPS}"

case "$ACTION_FUSION_METHOD" in
  project|concat) ;;
  *) echo "ERROR: --action-fusion must be project or concat" >&2; exit 1 ;;
esac
case "$SAVE_OPTIMIZER_STATE" in
  true|false|True|False) ;;
  *) echo "ERROR: --save-optimizer-state must be true or false" >&2; exit 1 ;;
esac
case "$AUTORESUME" in
  true|false|True|False) ;;
  *) echo "ERROR: --autoresume must be true or false" >&2; exit 1 ;;
esac

[[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: train JSONL not found: $TRAIN_JSONL" >&2; exit 1; }
[[ -f "$ACTION_SCALE_FILE" ]] || { echo "ERROR: action bins CSV not found: $ACTION_SCALE_FILE" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH" ]] || { echo "ERROR: tokenizer not found: $TOKENIZER_PATH" >&2; exit 1; }
[[ -e "$VQGAN_CKPT" ]] || { echo "ERROR: VQGAN checkpoint not found: $VQGAN_CKPT" >&2; exit 1; }
[[ -e "$LAPA_PARAMS" ]] || { echo "ERROR: initial LAPA params not found: $LAPA_PARAMS" >&2; exit 1; }

ACTION_VOCAB_SIZE="$(head -1 "$ACTION_SCALE_FILE" | awk -F, '{print NF}')"
WANDB_DIR="$OUTPUT_DIR/$EXPERIMENT_ID/wandb"

echo "[train-depth] root: $LAPA_ROOT"
echo "[train-depth] suite: $SUITE"
echo "[train-depth] model: $STAGE25_MODEL_NAME"
echo "[train-depth] fusion: $ACTION_FUSION_METHOD"
echo "[train-depth] train jsonl: $TRAIN_JSONL"
echo "[train-depth] image root: $IMAGE_ROOT"
echo "[train-depth] depth dir: $DEPTH_DATA_DIR"
echo "[train-depth] depth manifest: ${DEPTH_MANIFEST:-<none>}"
echo "[train-depth] action bins: $ACTION_SCALE_FILE"
echo "[train-depth] action vocab size: $ACTION_VOCAB_SIZE"
echo "[train-depth] init params: $LAPA_PARAMS"
echo "[train-depth] output: $OUTPUT_DIR/$EXPERIMENT_ID"
echo "[train-depth] total steps: $TOTAL_STEPS | batch size: $BATCH_SIZE | lr: $LR"
echo "[train-depth] save model freq: $SAVE_MODEL_FREQ"
echo "[train-depth] save milestone freq: $SAVE_MILESTONE_FREQ | keep last milestones: $KEEP_LAST_MILESTONES"
echo "[train-depth] save optimizer state: $SAVE_OPTIMIZER_STATE | autoresume: $AUTORESUME"

python_args=(
  -u -m latent_pretraining.train
  --modality="vision,action,delta"
  --mesh_dim="$MESH_DIM"
  --dtype="bf16"
  --total_steps="$TOTAL_STEPS"
  --log_freq="${LOG_FREQ:-1}"
  --eval_steps="${EVAL_STEPS:-0}"
  --save_model_freq="$SAVE_MODEL_FREQ"
  --eval_log_freq="${EVAL_LOG_FREQ:-100}"
  --save_milestone_freq="$SAVE_MILESTONE_FREQ"
  --keep_last_milestones="$KEEP_LAST_MILESTONES"
  --runtime_log_steps="${RUNTIME_LOG_STEPS:-3}"
  --abort_on_nonfinite=True
  --diagnose_numerics="${DIAGNOSE_NUMERICS:-False}"
  --load_llama_config="7b"
  --load_checkpoint="params::$LAPA_PARAMS"
  --update_llama_config="dict(action_vocab_size=${ACTION_VOCAB_SIZE},depth_feature_dim=1024,action_fusion_method='${ACTION_FUSION_METHOD}',delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)"
  --tokenizer.vocab_file="$TOKENIZER_PATH"
  --optimizer.type="adamw"
  --llama.action_vocab_size="$ACTION_VOCAB_SIZE"
  --llama.delta_vocab_size=8
  --optimizer.accumulate_gradient_steps=1
  --optimizer.adamw_optimizer.weight_decay=0
  --optimizer.adamw_optimizer.lr="$LR"
  --optimizer.adamw_optimizer.end_lr="$LR"
  --optimizer.adamw_optimizer.lr_warmup_steps=0
  --optimizer.adamw_optimizer.lr_decay_steps="$TOTAL_STEPS"
  --freeze_vision_params=True
  --use_data_sharded_loader=True
  --train_dataset.type="json_vision_delta_action"
  --train_dataset.delta_vision_action_processor.fields_from_example="fields"
  --train_dataset.delta_vision_action_processor.sample_id_key="${JSON_ID_KEY:-id}"
  --train_dataset.delta_vision_action_processor.sample_id_source="${JSON_ID_SOURCE:-auto}"
  --train_dataset.delta_vision_action_processor.n_tokens_per_action=7
  --train_dataset.delta_vision_action_processor.n_tokens_per_delta=4
  --train_dataset.delta_vision_action_processor.img_aug=True
  --train_dataset.delta_vision_action_processor.vqgan_checkpoint_path="$VQGAN_CKPT"
  --train_dataset.delta_vision_action_processor.image_absolute_path="$IMAGE_ROOT"
  --train_dataset.delta_vision_action_processor.max_n_frames=1
  --train_dataset.json_delta_action_dataset.mode="pad"
  --train_dataset.json_delta_action_dataset.path="$TRAIN_JSONL"
  --train_dataset.json_delta_action_dataset.seq_length="$SEQ_LENGTH"
  --train_dataset.json_delta_action_dataset.batch_size="$BATCH_SIZE"
  --train_dataset.json_delta_action_dataset.tokenizer_processes="${TOKENIZER_PROCESSES:-1}"
  --train_dataset.json_delta_action_dataset.tokenizer_parallel_chunk_size="${TOKENIZER_PARALLEL_CHUNK_SIZE:-128}"
  --train_dataset.json_delta_action_dataset.tokenizer_parallel_batch_size="${TOKENIZER_PARALLEL_BATCH_SIZE:-128}"
  --train_dataset.json_delta_action_dataset.use_data_sharded_loader=True
  --train_dataset.json_delta_action_dataset.depth_feature_data_dir="$DEPTH_DATA_DIR"
  --train_dataset.json_delta_action_dataset.depth_feature_key="${DEPTH_FEATURE_KEY:-auto}"
  --train_dataset.json_delta_action_dataset.depth_feature_id_key="${DEPTH_ID_KEY:-auto}"
  --train_dataset.json_delta_action_dataset.depth_feature_dim=1024
  --checkpointer.save_optimizer_state="$SAVE_OPTIMIZER_STATE"
  --autoresume="$AUTORESUME"
  --logger.append_uuid=False
  --logger.online="$WANDB_ONLINE"
  --logger.project_id="${PROJECT_ID:-depth_policy}"
  --logger.experiment_id="$EXPERIMENT_ID"
  --logger.experiment_note="${EXPERIMENT_NOTE:-depth-aware policy fine-tuning}"
  --logger.output_dir="$OUTPUT_DIR"
  --logger.wandb_dir="$WANDB_DIR"
)

if [[ -n "$DEPTH_MANIFEST" ]]; then
  python_args+=(--train_dataset.json_delta_action_dataset.depth_feature_manifest="$DEPTH_MANIFEST")
fi

env -i \
  HOME="$HOME" \
  USER="${USER:-}" \
  PATH="$PATH" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  PYTHONPATH="$LAPA_ROOT:${PYTHONPATH:-}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}" \
  TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}" \
  JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}" \
  WANDB_MODE="${WANDB_MODE:-offline}" \
  LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true}" \
  python3 "${python_args[@]}"

echo "[train-depth] params: $OUTPUT_DIR/$EXPERIMENT_ID/streaming_params"
echo "[train-depth] train state: $OUTPUT_DIR/$EXPERIMENT_ID/streaming_train_state"
