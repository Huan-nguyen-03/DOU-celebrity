#!/usr/bin/env python3
"""
Concept Inversion (Textual Inversion) attack training for identity unlearning.

White-box red-team protocol (Pham et al. / Textual Inversion style):
  1. Load the *unlearned* model (SD1.4 + identity LoRA), freeze weights.
  2. Introduce a placeholder token S* and learn only its text embedding
     on a small gallery of target-identity images (e.g. Barack Obama).
  3. At inference, prompts like "a photo of S*" try to reconstruct the identity.

This does NOT train the DUO LoRA; it attacks a fixed unlearned checkpoint.

Usage:
  python -m eval.train_concept_inversion \\
      --train_data_dir datasets/galleries/Barack_Obama \\
      --lora_path train/outputs/.../Identity_Obama/checkpoint-500 \\
      --placeholder_token "S*" \\
      --output_dir outputs/ci_obama \\
      --max_train_steps 1500

Outputs:
  output_dir/
    learned_embeds.bin          # {token: embedding tensor}
    learned_embeds.safetensors  # same, if safetensors available
    tokenizer/                  # tokenizer with new token
    train_config.json
    samples/                    # optional mid/final sample grids
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from eval.sd_utils import find_lora_dir, free_mem, generate_image

IMAGENET_TEMPLATES_SMALL = [
    "a photo of {}",
    "a portrait of {}",
    "a close-up photo of {}",
    "a photo of the face of {}",
    "a realistic photo of {}",
    "a cropped photo of {}",
    "the photo of {}",
    "a bright photo of {}",
    "a dark photo of {}",
    "a rendering of {}",
    "a painting of {}",
    "a photo of a {}",
    "a photo of my {}",
    "a photo of one {}",
    "a good photo of {}",
    "a close-up of {}",
    "a rendition of {}",
    "a photo of the {}",
    "a portrait photo of {}",
    "an official portrait of {}",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Concept Inversion (textual inversion) on unlearned SD+LoRA"
    )
    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    p.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Unlearned identity LoRA dir (recommended for attack-on-unlearn protocol)",
    )
    p.add_argument(
        "--no_lora",
        action="store_true",
        help="Train CI on base SD only (no unlearn LoRA)",
    )
    p.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Folder of target identity images (jpg/png)",
    )
    p.add_argument(
        "--placeholder_token",
        type=str,
        default="<obama-ci>",
        help="Pseudo-word token to learn (Concept Inversion token). Prefer a single rare token like <obama-ci>.",
    )
    p.add_argument(
        "--initializer_token",
        type=str,
        default="person",
        help="Token whose embedding initializes the placeholder",
    )
    p.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        choices=["object", "style"],
        help="Prompt template family (object ≈ identity/person)",
    )
    p.add_argument("--output_dir", type=str, default="./outputs/ci_train")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=1500)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--lr_scheduler", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_vectors", type=int, default=1, help="Multi-vector TI (usually 1)")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--sample_steps", type=int, default=500)
    p.add_argument("--num_sample_images", type=int, default=4)
    p.add_argument("--sample_prompt", type=str, default=None)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="CFG rescale for mid-train samples (0 = off). Use 0.7 for FaceInpaint protocol.",
    )
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Repeat each image this many times in the dataset epoch construction",
    )
    return p.parse_args(argv)


def list_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG"):
        paths.extend(folder.glob(ext))
    paths = sorted({p.resolve() for p in paths})
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    return paths


def save_progress(
    text_encoder,
    placeholder_tokens: Sequence[str],
    placeholder_token_ids: Sequence[int],
    save_path: Path,
) -> None:
    import torch

    save_path.parent.mkdir(parents=True, exist_ok=True)
    learned = text_encoder.get_input_embeddings().weight
    embeds = {
        token: learned[tid].detach().cpu().clone()
        for token, tid in zip(placeholder_tokens, placeholder_token_ids)
    }
    # single-token dict form used by diffusers load_textual_inversion
    if len(embeds) == 1:
        payload = embeds
    else:
        # multi-vector: store as token -> stacked or dict of subtokens
        payload = embeds

    torch.save(payload, save_path.with_suffix(".bin"))
    try:
        from safetensors.torch import save_file

        flat = {k: v for k, v in payload.items()}
        save_file(flat, str(save_path.with_suffix(".safetensors")))
    except Exception as e:  # noqa: BLE001
        print("safetensors save skip:", e)
    print("Saved embeddings:", save_path.with_suffix(".bin"))


def _module_device(module):
    return next(module.parameters()).device


def _cast_pipe_fp32(pipe):
    """
    Mixed-precision train leaves UNet/VAE in fp16. SD inference builds
    timestep embeddings in fp32, which then crashes:
      RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
    Cast the denoising stack to fp32 for sampling, then restore.
    """
    import torch

    snapshot = {}
    for name in ("unet", "vae", "text_encoder"):
        mod = getattr(pipe, name, None)
        if mod is None:
            continue
        try:
            snapshot[name] = next(mod.parameters()).dtype
        except StopIteration:
            continue
        mod.to(dtype=torch.float32)
    return snapshot


def _restore_pipe_dtypes(pipe, snapshot) -> None:
    for name, dtype in snapshot.items():
        mod = getattr(pipe, name, None)
        if mod is not None:
            mod.to(dtype=dtype)


def sample_images(
    pipe,
    prompt: str,
    out_dir: Path,
    n: int,
    seed: int,
    steps: int = 25,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _module_device(pipe.unet)
    snapshot = _cast_pipe_fp32(pipe)
    try:
        for i in range(n):
            img = generate_image(
                pipe,
                prompt,
                int(seed) + i,
                steps,
                guidance_scale,
                guidance_rescale=guidance_rescale,
            )
            img.save(out_dir / f"sample_{i:02d}.png")
        print(f"Wrote {n} samples to {out_dir}")
    finally:
        _restore_pipe_dtypes(pipe, snapshot)
        if device.type == "cuda":
            pipe.unet.to(device)
            pipe.vae.to(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = out_dir / "logs"
    accelerator_project_config = ProjectConfiguration(
        project_dir=str(out_dir), logging_dir=str(logging_dir)
    )
    mixed = None if args.mixed_precision == "no" else args.mixed_precision
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed,
        project_config=accelerator_project_config,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ---- tokenizer + placeholder tokens ----
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    placeholder_tokens = [args.placeholder_token]
    additional = []
    for i in range(1, args.num_vectors):
        additional.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional

    num_added = tokenizer.add_tokens(placeholder_tokens)
    if num_added != len(placeholder_tokens):
        # token may already exist; still proceed if single known token
        print(
            f"WARNING: add_tokens returned {num_added}, expected {len(placeholder_tokens)}. "
            "If the placeholder already exists in vocab, pick another string."
        )

    token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    if any(tid == tokenizer.unk_token_id for tid in token_ids):
        raise ValueError(
            f"Placeholder token(s) mapped to UNK: {placeholder_tokens}. "
            "Choose a rare multi-char token like 'S*' or '<obama-ci>'."
        )
    print("placeholder tokens:", placeholder_tokens, "ids:", token_ids)

    # ---- models ----
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder.resize_token_embeddings(len(tokenizer))

    # init placeholder embed from initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    init_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    if len(init_ids) != 1:
        raise ValueError(
            f"initializer_token must be a single token, got {args.initializer_token!r} -> {init_ids}"
        )
    init_id = init_ids[0]
    with torch.no_grad():
        for tid in token_ids:
            token_embeds[tid] = token_embeds[init_id].clone()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # freeze all except placeholder embedding rows
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    # Load unlearn LoRA onto UNet (and optionally text encoder — DUO uses UNet LoRA)
    lora_dir = None
    if not args.no_lora:
        if not args.lora_path:
            raise ValueError("Provide --lora_path or pass --no_lora")
        lora_dir = find_lora_dir(args.lora_path)
        # Use pipeline helper to load LoRA then extract unet
        tmp_pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            safety_checker=None,
            requires_safety_checker=False,
        )
        tmp_pipe.load_lora_weights(lora_dir)
        # fuse for simpler frozen attack target (optional but stable)
        try:
            tmp_pipe.fuse_lora()
            print("Fused LoRA into UNet for CI training")
        except Exception as e:  # noqa: BLE001
            print("fuse_lora skipped:", e)
        unet = tmp_pipe.unet
        text_encoder = tmp_pipe.text_encoder
        vae = tmp_pipe.vae
        del tmp_pipe
        free_mem()
        # re-freeze after fuse
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder.get_input_embeddings().requires_grad_(True)

    # Only optimize input embeddings
    text_encoder.get_input_embeddings().requires_grad_(True)
    params_to_optimize = text_encoder.get_input_embeddings().parameters()

    # ---- dataset ----
    img_paths = list_images(Path(args.train_data_dir))
    print(f"Gallery images: {len(img_paths)} from {args.train_data_dir}")

    templates = IMAGENET_TEMPLATES_SMALL
    placeholder_string = " ".join(placeholder_tokens)

    class CIDataset(Dataset):
        def __init__(self):
            self.paths = img_paths * max(1, args.repeats)
            self.tf = transforms.Compose(
                [
                    transforms.Resize(
                        args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx % len(img_paths)]
            image = Image.open(path).convert("RGB")
            if random.random() < 0.5:
                try:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                except Exception:  # noqa: BLE001
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
            pixel_values = self.tf(image)
            template = random.choice(templates)
            text = template.format(placeholder_string)
            input_ids = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataset = CIDataset()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # keep original embeddings reference for non-placeholder freeze
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )

    # config dump
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["lora_dir_resolved"] = lora_dir
    cfg["n_gallery_images"] = len(img_paths)
    cfg["placeholder_token_ids"] = token_ids
    (out_dir / "train_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("***** Concept Inversion training *****")
    print(f"  steps={args.max_train_steps}  lr={args.learning_rate}  token={args.placeholder_token!r}")
    print(f"  lora={lora_dir}  gallery_n={len(img_paths)}")

    global_step = 0
    t0 = time.time()
    progress = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="ci-train",
    )

    text_encoder.train()
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(text_encoder):
                # encode images
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(
                    dtype=weight_dtype
                )
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(noise_scheduler.config.prediction_type)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # freeze all token embeds except placeholder ids
                with torch.no_grad():
                    emb = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
                    index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
                    for tid in token_ids:
                        index_no_updates[tid] = False
                    emb[index_no_updates] = orig_embeds_params[index_no_updates]

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                progress.set_postfix(loss=float(loss.detach().item()))

                if (global_step % args.save_steps == 0 or global_step == args.max_train_steps) and accelerator.is_main_process:
                    save_progress(
                        accelerator.unwrap_model(text_encoder),
                        placeholder_tokens,
                        token_ids,
                        out_dir / f"learned_embeds-steps-{global_step}",
                    )
                    # also write canonical name
                    save_progress(
                        accelerator.unwrap_model(text_encoder),
                        placeholder_tokens,
                        token_ids,
                        out_dir / "learned_embeds",
                    )

                if (
                    args.sample_steps > 0
                    and (global_step % args.sample_steps == 0 or global_step == args.max_train_steps)
                    and accelerator.is_main_process
                ):
                    prompt = args.sample_prompt or f"a photo of {args.placeholder_token}"
                    pipe = StableDiffusionPipeline(
                        vae=vae,
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        unet=unet,
                        scheduler=noise_scheduler,
                        safety_checker=None,
                        feature_extractor=None,
                        requires_safety_checker=False,
                    )
                    pipe.set_progress_bar_config(disable=True)
                    try:
                        sample_images(
                            pipe,
                            prompt,
                            out_dir / "samples" / f"step_{global_step}",
                            args.num_sample_images,
                            args.seed,
                            guidance_scale=args.guidance_scale,
                            guidance_rescale=args.guidance_rescale,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"WARNING: mid-train sampling failed at step {global_step}: {e}. "
                            "Embeddings were still saved."
                        )
                    del pipe
                    free_mem()
                    # restore train dtypes after fp32 sampling
                    unet.to(accelerator.device, dtype=weight_dtype)
                    vae.to(accelerator.device, dtype=weight_dtype)
                    text_encoder.train()

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_progress(
            accelerator.unwrap_model(text_encoder),
            placeholder_tokens,
            token_ids,
            out_dir / "learned_embeds",
        )
        tokenizer.save_pretrained(out_dir / "tokenizer")

    minutes = (time.time() - t0) / 60.0
    result = {
        "output_dir": str(out_dir),
        "placeholder_token": args.placeholder_token,
        "placeholder_token_ids": token_ids,
        "lora_dir": lora_dir,
        "max_train_steps": args.max_train_steps,
        "minutes": minutes,
        "learned_embeds": str(out_dir / "learned_embeds.bin"),
    }
    (out_dir / "train_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("CI training done:", result)
    return result


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
