#!/usr/bin/env bash
# MS-COCO utility metrics (FID / CLIP / LPIPS) for identity LoRA prior check.
#
#   bash scripts/eval-coco-metrics.sh \
#       --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
#       --num_samples 1000

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_OUT="${ROOT}/outputs/coco_eval"
ARGS=("$@")
has_out=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  if [[ "$a" == "--output_dir" || "$a" == --output_dir=* ]]; then
    has_out=1
    break
  fi
done
if [[ $has_out -eq 0 ]]; then
  ARGS+=(--output_dir "$DEFAULT_OUT")
fi

echo "[eval-coco-metrics] python -m eval.coco_metrics ${ARGS[*]}"
python -m eval.coco_metrics "${ARGS[@]}"
