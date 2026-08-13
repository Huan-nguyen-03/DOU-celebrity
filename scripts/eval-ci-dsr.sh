#!/usr/bin/env bash
# Concept Inversion DSR eval (gpt-4o-mini via LiteLLM).
#
# Requires: LITELLM__URL, LITELLM__TOKEN (unless --skip_vlm)
#
# Example:
#   bash scripts/eval-ci-dsr.sh \
#       --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
#       --learned_embeds outputs/ci_obama/learned_embeds.bin

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_OUT="${ROOT}/outputs/ci_dsr_eval"
ARGS=("$@")
has_out=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  if [[ "$a" == "--output_dir" || "$a" == --output_dir=* ]]; then has_out=1; fi
done
if [[ $has_out -eq 0 ]]; then
  ARGS+=(--output_dir "$DEFAULT_OUT")
fi

echo "[eval-ci-dsr] python -m eval.ci_dsr_eval ${ARGS[*]}"
python -m eval.ci_dsr_eval "${ARGS[@]}"
