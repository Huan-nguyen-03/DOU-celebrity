#!/usr/bin/env bash
# Train Concept Inversion (textual inversion) on unlearned identity LoRA.
#
# Example:
#   bash scripts/train-concept-inversion.sh \
#       --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
#       --train_data_dir datasets/galleries/Barack_Obama \
#       --output_dir outputs/ci_obama

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_DATA="${ROOT}/datasets/galleries/Barack_Obama"
DEFAULT_OUT="${ROOT}/outputs/ci_obama"
ARGS=("$@")

has_data=0
has_out=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  if [[ "$a" == "--train_data_dir" || "$a" == --train_data_dir=* ]]; then has_data=1; fi
  if [[ "$a" == "--output_dir" || "$a" == --output_dir=* ]]; then has_out=1; fi
done
if [[ $has_data -eq 0 ]]; then
  ARGS+=(--train_data_dir "$DEFAULT_DATA")
fi
if [[ $has_out -eq 0 ]]; then
  ARGS+=(--output_dir "$DEFAULT_OUT")
fi

echo "[train-ci] python -m eval.train_concept_inversion ${ARGS[*]}"
python -m eval.train_concept_inversion "${ARGS[@]}"
