#!/usr/bin/env bash
# Full identity eval = ArcFace suites (+ RAB/Sneaky analogues) + COCO utility.
#
#   bash scripts/eval-full-identity.sh \
#     --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
#     --gallery_dir /path/to/obama_photos \
#     --coco_num_samples 1000

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_OUT="${ROOT}/outputs/full_identity_eval"
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

echo "[eval-full-identity] python -m eval.run_full_identity_eval ${ARGS[*]}"
python -m eval.run_full_identity_eval "${ARGS[@]}"
