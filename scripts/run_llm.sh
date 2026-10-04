#!/usr/bin/env bash
# Retained GSM8K configurations; model/data paths are supplied by the user.
# Usage: MODEL_PATH=/path/to/model DATA_ROOT=/path/to/data bash scripts/run_llm.sh
#        [llama8b|qwen9b] [clients] [seed] [rounds] [gpu] [config overrides ...]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# shellcheck source=env.sh
source "$HERE/env.sh"
: "${MODEL_PATH:?Set MODEL_PATH to the local backbone directory or model identifier}"
BACKBONE=${1:-llama8b}
case "$BACKBONE" in
  llama8b|qwen9b) CONFIG="configs/barylora_gsm8k_${BACKBONE}.yaml" ;;
  *) printf '%s\n' 'Backbone must be llama8b or qwen9b' >&2; exit 2 ;;
esac
N=${2:-3}
SEED=${3:-42}
ROUNDS=${4:-100}
GPU=${5:-0}
PY=${ENV_PY:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/llm}
MODEL_CACHE=${MODEL_CACHE:-$HF_HOME/hub}
OUT=${OUT:-$ROOT/outputs/gsm8k/${BACKBONE}_n${N}_seed${SEED}_$(date -u +%Y%m%dT%H%M%S)}
EXTRA_ARGS=("${@:6}")
mkdir -p "$OUT/ckpts" "$OUT/out"
cd "$ROOT"
"$PY" federatedscope/main.py --cfg "$CONFIG" \
  seed "$SEED" use_gpu True device "$GPU" eval_device "$GPU" \
  model.type "$MODEL_PATH@huggingface_llm" data.root "$DATA_ROOT" \
  llm.cache.model "$MODEL_CACHE" \
  federate.client_num "$N" federate.sample_client_num "$N" \
  federate.total_round_num "$ROUNDS" \
  federate.save_to "$OUT/ckpts/barylora_gsm8k_${BACKBONE}_n${N}_seed${SEED}.ckpt" \
  outdir "$OUT/out" "${EXTRA_ARGS[@]}"
