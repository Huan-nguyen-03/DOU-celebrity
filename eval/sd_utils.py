#!/usr/bin/env python3
"""Shared SD pipeline helpers for identity + COCO eval."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional


def free_mem() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_lora_dir(path: str) -> str:
    """Locate directory that contains pytorch_lora_weights.safetensors."""
    p = Path(path)
    candidates: list[Path] = [p]
    if p.exists() and p.is_dir():
        for c in sorted(p.glob("checkpoint-*")):
            candidates.append(c)
        for name in ("checkpoint-1000", "checkpoint-500", "checkpoint-250"):
            candidates.append(p / name)
    for c in candidates:
        if (c / "pytorch_lora_weights.safetensors").exists():
            print(f"  LoRA found: {c}")
            return str(c)
    if p.exists():
        hits = list(p.rglob("pytorch_lora_weights.safetensors"))
        if hits:
            print(f"  LoRA found (rglob): {hits[0].parent}")
            return str(hits[0].parent)
    raise FileNotFoundError(
        f"Cannot find pytorch_lora_weights.safetensors under: {path}"
    )


def build_sd_pipeline(
    base_model_id: str,
    device: str,
    dtype: Any,
    lora_path: Optional[str] = None,
    scheduler: str = "pndm",
):
    """
    Build SD1.4 pipeline, optionally with a single identity LoRA.

    scheduler:
      - "pndm": default SD1.4
      - "dpm": DPMSolverMultistep (faster)
    """
    from diffusers import (
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
    )

    print("Loading base model:", base_model_id)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    for fn_name in ("enable_attention_slicing", "enable_vae_slicing"):
        try:
            getattr(pipe, fn_name)()
        except Exception:  # noqa: BLE001
            pass

    if lora_path is not None:
        lora_dir = find_lora_dir(lora_path)
        pipe.load_lora_weights(lora_dir)
        print("Loaded LoRA from", lora_dir)
    else:
        print("Base model only (no LoRA)")

    return pipe
