#!/usr/bin/env bash
# Evaluate task:checkpoint pairs; options after -- are forwarded to eval_ckpt.py.
# Usage: bash infer/eval_all.sh OUT task:checkpoint [...] -- [inference options ...]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?usage: eval_all.sh OUT task:checkpoint [...] -- [inference options ...]}
shift
CHECKPOINTS=()
EXTRA_ARGS=()
while (( $# )); do
  if [[ "$1" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
    break
  fi
  CHECKPOINTS+=("$1")
  shift
done
if (( ${#CHECKPOINTS[@]} == 0 )); then
  printf '%s\n' 'Provide at least one task:checkpoint pair' >&2
  exit 2
fi
mkdir -p "$OUT"
PY=${ENV_PY:-python}
printf 'name\ttask\tmetric_name\tmetric_value\tckpt\tsha256\n' > "$OUT/summary.tsv"
for spec in "${CHECKPOINTS[@]}"; do
  task=${spec%%:*}
  ckpt=${spec#*:}
  name=$(basename -- "$ckpt" .ckpt)
  "$PY" "$HERE/eval_ckpt.py" --ckpt "$ckpt" --task "$task" \
    --tok-len 128 --batch-size "${BATCH_SIZE:-16}" --device "${DEVICE:-cpu}" \
    --out "$OUT/${name}.metrics.json" "${EXTRA_ARGS[@]}"
  "$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print('\t'.join([sys.argv[2], m['task'], m['metric_name'], str(m['metric_value']), m['checkpoint'], m['checkpoint_sha256']]))" \
    "$OUT/${name}.metrics.json" "$name" >> "$OUT/summary.tsv"
done
printf 'wrote %s\n' "$OUT/summary.tsv"
