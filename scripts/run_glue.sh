#!/usr/bin/env bash
# Clean single-task GLUE runner for BaryLoRA.
# Usage: bash scripts/run_glue.sh [task] [clients] [seed] [rounds] [gpu] [config overrides ...]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

# shellcheck source=env.sh
source "$HERE/env.sh"
PY=${ENV_PY:-python}
DATA_ROOT=${DATA_ROOT:-${HF_DATASETS_CACHE:-$HF_HOME/datasets}}
MODEL_CACHE=${MODEL_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE="$DATA_ROOT"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
export FEDLORA_GLUE_REPO=${FEDLORA_GLUE_REPO:-nyu-mll/glue}

TASK=${1:-mnli}
N=${2:-3}
SEED=${3:-42}
ROUNDS=${4:-250}
GPU=${5:-0}
OUT=${OUT:-$ROOT/outputs/glue/${TASK}_n${N}_seed${SEED}_$(date -u +%Y%m%dT%H%M%S)}
EXTRA_ARGS=("${@:6}")
mkdir -p "$OUT/ckpts" "$OUT/out"

cd "$ROOT"
"$PY" federatedscope/main.py \
  --cfg configs/barylora_glue.yaml \
  seed "$SEED" use_gpu True device "$GPU" eval_device "$GPU" \
  data.type "${TASK}@glue" data.root "$DATA_ROOT" \
  llm.cache.model "$MODEL_CACHE" \
  data.splitter lda data.splitter_args "[{'alpha': 0.5}]" \
  federate.client_num "$N" federate.sample_client_num "$N" \
  federate.total_round_num "$ROUNDS" \
  federate.save_to "$OUT/ckpts/barylora_${TASK}_n${N}_seed${SEED}.ckpt" \
  outdir "$OUT/out" \
  eval.metrics "['accuracy']" "${EXTRA_ARGS[@]}"
