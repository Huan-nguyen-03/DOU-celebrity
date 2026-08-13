# Barack Obama reference gallery

Public-figure photos for **Concept Inversion / Textual Inversion** research
(identity unlearning evaluation). **Not** for training the DUO unlearn LoRA itself.

## License / source

All images were downloaded from **Wikimedia Commons**. Most are:

- U.S. government works (White House / Pete Souza official duties) — **public domain** in the U.S., or
- Freely licensed portraits (e.g. CC BY) as labeled on Commons.

Always re-check the file page on Commons if you redistribute.

## Files

**21 unique** images, numbered `01.jpg` … `21.jpg` after dedupe/normalize
(Wikimedia Commons: official portraits, White House photos, public-domain
portraits — see Commons file pages for exact titles/licenses).

## Usage

```bash
python -m eval.train_concept_inversion \
  --train_data_dir datasets/galleries/Barack_Obama \
  --lora_path /path/to/Identity_Obama/checkpoint-500 \
  --placeholder_token "S*" \
  --output_dir outputs/ci_obama
```
