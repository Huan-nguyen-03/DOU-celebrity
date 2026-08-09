#!/usr/bin/env bash
# Ring-A-Bell identity red-team with VLM DSR (gpt-4o-mini via LiteLLM).
# No ArcFace / E1 / E2 / E4.
#
# Requires env or secrets:
#   LITELLM__URL, LITELLM__TOKEN
#
# Example:
#   export LITELLM__URL=...
#   export LITELLM__TOKEN=...
#   bash scripts/eval-rab-dsr.sh \
#       --lora_path train/outputs/unlearn/SD-train/dpo/500/Identity_Obama/checkpoint-500 \
#       --identity_name "Barack Obama" \
#       --rab_prompt_file eval/ring_a_bell_prompts/Obama_1.5_length_10.txt

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_OUT="${ROOT}/outputs/rab_dsr_eval"
DEFAULT_RAB="${ROOT}/eval/ring_a_bell_prompts/Obama_1.5_length_10.txt"
ARGS=("$@")

has_out=0
has_rab=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  if [[ "$a" == "--output_dir" || "$a" == --output_dir=* ]]; then
    has_out=1
  fi
  if [[ "$a" == "--rab_prompt_file" || "$a" == --rab_prompt_file=* ]]; then
    has_rab=1
  fi
done
if [[ $has_out -eq 0 ]]; then
  ARGS+=(--output_dir "$DEFAULT_OUT")
fi
if [[ $has_rab -eq 0 ]]; then
  ARGS+=(--rab_prompt_file "$DEFAULT_RAB")
fi

echo "[eval-rab-dsr] python -m eval.rab_dsr_eval ${ARGS[*]}"
python -m eval.rab_dsr_eval "${ARGS[@]}"
