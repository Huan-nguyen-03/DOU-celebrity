#!/usr/bin/env bash
# Identity (celebrity) unlearning eval — ArcFace ISR / IER.
# NOT nude/violence. Use this after training Identity_Obama LoRA.
#
# Example:
#   bash scripts/eval-identity.sh \
#       --lora_path train/outputs/unlearn/SD-train/dpo/500/Identity_Obama/checkpoint-500 \
#       --identity_name "Barack Obama" \
#       --gallery_dir /path/to/obama_photos \
#       --num_seeds 5

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_OUT="${ROOT}/outputs/identity_eval"
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

echo "[eval-identity] python -m eval.identity_metrics ${ARGS[*]}"
python -m eval.identity_metrics "${ARGS[@]}"
