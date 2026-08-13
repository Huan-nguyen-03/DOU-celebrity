# DUO: Direct Unlearning Optimization for Robust and Safe Text-to-Image Models

This repository contains the official implementation of [Direct Unlearning Optimization for Robust and Safe Text-to-Image Models](https://arxiv.org/abs/2407.21035).


**Our work is accepted at :tada:NeurIPS 2024:tada:!** This repository code is based on the settings used for the NeurIPS submission and differs from the configuration in the arXiv paper. The NeurIPS version of the paper will be uploaded to arXiv soon.

## Setup

### Prerequisites

- Docker
- NVIDIA GPU with CUDA support (4-GPU is recommended)

### Environment Setup

1. Use the following Docker container image:
   ```
   nvcr.io/nvidia/pytorch:23.12-py3
   ```

2. After setting up the Docker container, run:
   ```bash
   git clone https://github.com/naver-ai/DUO
   cd duo 
   pip install -r requirements.txt
   accelerate config
   ```

## Training

### Dataset preparation
```bash
bash scripts/prepare-dataset.sh
```

### Stable Diffusion 1.4v
#### Nudity Experiment
```bash
bash scripts/sd-nudity.sh
```

#### Violence Experiment
```bash
bash scripts/sd-violence.sh
```

- We use `train_batch_size=4`. If the number of GPUs is not equal to 4, please adjust the batch size accordingly.
- Specify the `save_dir` in all scripts. This is where the unlearned LoRA weights and evaluation results will be stored.
- For the violence experiment, provide an OpenAI API key (refer to `scripts/sd-violence-figures.sh`).

### Stable Diffusion 3

```bash
bash scripts/sd3-nudity.sh
```

## Inference
We provide simple inference codes in `inference.py`.
```bash
python3 inference.py --prompt "a naked woman, realistic photo" --exp_type "nudity" --unlearn_model_path "train/outputs/unlearn/SD-train/dpo/500"
python3 inference.py --prompt "a man bleeding" --exp_type "violence" --unlearn_model_path "train/outputs/unlearn/SD-train/dpo/1000"
```

## Acknowledgements

This repository is heavily based on the following projects:
- [Diffusers](https://github.com/huggingface/diffusers)
- [DCO](https://github.com/kyungmnlee/dco)
- [LoRA](https://github.com/cloneofsimo/lora/tree/master)

## License
```
DUO
Copyright (c) 2024-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
# DUO-celebrity

---

## Fork note — DUO_celebrity (identity / celebrity)

Research fork for **celebrity / identity unlearning** (e.g. Barack Obama).
Upstream train/data/infer scripts are unchanged.

Eval stack:

| Eval | Module | Metric |
|------|--------|--------|
| **Red-team (Ring-A-Bell)** | `eval.rab_dsr_eval` | DSR / ASR via gpt-4o-mini |
| **Red-team (Concept Inversion)** | `eval.train_concept_inversion` + `eval.ci_dsr_eval` | TI token + DSR |
| **Prior utility (MS-COCO)** | `eval.coco_metrics` | FID / CLIP / LPIPS |

Obama reference gallery (Wikimedia, research-only):  
`datasets/galleries/Barack_Obama/` (+ `SOURCES.md`).

### 1. Ring-A-Bell + VLM DSR

```bash
export LITELLM__URL=...
export LITELLM__TOKEN=...

python -m eval.rab_dsr_eval \
  --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
  --identity_name "Barack Obama" \
  --rab_prompt_file eval/ring_a_bell_prompts/Obama_1.5_length_10.txt \
  --output_dir ./outputs/rab_dsr_eval

bash scripts/eval-rab-dsr.sh --lora_path ...
```

- **DSR ↑** = images judged *not* to show the identity  
- Kaggle: `duo-obama-redteam-rab.ipynb`

### 2. Concept Inversion (Textual Inversion)

Train a soft token on the **unlearned** model using the Obama gallery, then
generate / score DSR:

```bash
# Train CI token on unlearn LoRA
python -m eval.train_concept_inversion \
  --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
  --train_data_dir datasets/galleries/Barack_Obama \
  --placeholder_token "<obama-ci>" \
  --output_dir outputs/ci_obama \
  --max_train_steps 1500

bash scripts/train-concept-inversion.sh --lora_path ...

# DSR with gpt-4o-mini
python -m eval.ci_dsr_eval \
  --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
  --learned_embeds outputs/ci_obama/learned_embeds.bin \
  --placeholder_token "<obama-ci>" \
  --identity_name "Barack Obama" \
  --output_dir outputs/ci_dsr_eval

bash scripts/eval-ci-dsr.sh --lora_path ... --learned_embeds outputs/ci_obama/learned_embeds.bin
```

- Kaggle: `duo-obama-concept-inversion.ipynb` (train + optional DSR)

### 3. MS-COCO prior utility

```bash
python -m eval.coco_metrics \
  --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \
  --exp_type identity \
  --num_samples 1000 \
  --output_dir ./outputs/coco_eval

bash scripts/eval-coco-metrics.sh --lora_path ... --num_samples 1000
```

**Deps:** `openai>=1.0.0` + LiteLLM for VLM DSR; `torchmetrics[image]`, `matplotlib`, `scipy`, `lpips` for COCO.

