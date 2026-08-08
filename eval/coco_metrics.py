#!/usr/bin/env python3
# DUO_celebrity
# COCO utility metrics for unlearned T2I models (FID / CLIP / LPIPS).
#
# Reproduces the utility evaluation protocol described in:
#   "Direct Unlearning Optimization for Robust and Safe Text-to-Image Models"
#   (NeurIPS 2024, arXiv:2407.21035) §4.1 — MS-COCO validation FID / CLIP,
#   and LPIPS between base and unlearned generations with matched noise.
#
# Usage (from repo root):
#   python -m eval.coco_metrics \
#       --lora_path /path/to/Identity_Obama/checkpoint-500 \
#       --num_samples 1000 \
#       --output_dir ./outputs/coco_eval

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Heavy deps (torch / PIL / tqdm) are imported lazily inside functions so that
# `python -m eval.coco_metrics --help` works in minimal environments.


# ---------------------------------------------------------------------------
# Paper reference numbers (Table 1 / 2)
# ---------------------------------------------------------------------------
# Paper Table 1 SD1.4 utility (COCO) — reference only
PAPER_REF = {
    "SD_1.4": {"FID": 13.52, "CLIP": 30.95},
}


# ---------------------------------------------------------------------------
# Utilities (shared)
# ---------------------------------------------------------------------------
from eval.sd_utils import build_sd_pipeline, find_lora_dir, free_mem  # noqa: E402


def generate_images(
    pipe,
    captions: Sequence[str],
    out_dir: Path,
    seed_base: int,
    resume: bool,
    tag: str,
    device: str,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
) -> List[Path]:
    """Generate 1 image per caption; seed = seed_base + index."""
    import torch
    from tqdm.auto import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    t0 = time.time()
    with torch.inference_mode():
        for i, cap in enumerate(tqdm(captions, desc=f"Generate[{tag}]")):
            out_path = out_dir / f"{i:06d}.png"
            paths.append(out_path)
            if resume and out_path.exists():
                continue
            generator = torch.Generator(device=device).manual_seed(
                seed_base + i
            )
            img = pipe(
                prompt=cap,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=image_size,
                width=image_size,
                generator=generator,
            ).images[0]
            img.save(out_path)
            if (i + 1) % 50 == 0:
                free_mem()
    dt = (time.time() - t0) / 60.0
    print(
        f"[{tag}] {len(paths)} images -> {out_dir} ({dt:.1f} min wall)"
    )
    return paths


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_clip_score(
    image_paths: Sequence[Path],
    captions: Sequence[str],
    device: str,
    batch_size: int = 32,
) -> float:
    import torch
    from PIL import Image
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision import transforms
    from tqdm.auto import tqdm

    assert len(image_paths) == len(captions)
    metric = CLIPScore(
        model_name_or_path="openai/clip-vit-base-patch32"
    ).to(device)

    with torch.inference_mode():
        for i in tqdm(
            range(0, len(image_paths), batch_size), desc="CLIPScore"
        ):
            batch_paths = image_paths[i : i + batch_size]
            batch_caps = list(captions[i : i + batch_size])
            imgs = []
            for p in batch_paths:
                im = Image.open(p).convert("RGB").resize(
                    (224, 224), Image.BICUBIC
                )
                imgs.append(transforms.functional.pil_to_tensor(im))
            imgs_t = torch.stack(imgs).to(device)
            metric.update(imgs_t, batch_caps)

        val = float(metric.compute().detach().cpu())
    del metric
    free_mem()
    return val


def compute_fid(
    paths_real: Sequence[Path],
    paths_fake: Sequence[Path],
    device: str,
    batch_size: int = 32,
) -> float:
    import torch
    from PIL import Image
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms
    from tqdm.auto import tqdm

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(
        device
    )
    tf = transforms.Compose(
        [
            transforms.Resize(
                (299, 299),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )

    n = min(len(paths_real), len(paths_fake))
    if n < 2:
        raise ValueError(
            f"Not enough images for FID: real={len(paths_real)} "
            f"fake={len(paths_fake)}"
        )
    pr = list(paths_real)[:n]
    pf = list(paths_fake)[:n]
    print(f"FID n={n}")

    def feed(paths: Sequence[Path], real: bool) -> None:
        tag = "real" if real else "fake"
        for i in tqdm(
            range(0, len(paths), batch_size), desc=f"FID-{tag}"
        ):
            batch = [
                tf(Image.open(p).convert("RGB"))
                for p in paths[i : i + batch_size]
            ]
            x = torch.stack(batch).to(device)
            fid.update(x, real=real)

    with torch.inference_mode():
        feed(pr, True)
        feed(pf, False)
        val = float(fid.compute().detach().cpu())
    del fid
    free_mem()
    return val


def compute_lpips_paired(
    paths_a: Sequence[Path],
    paths_b: Sequence[Path],
    device: str,
    image_size: int,
    batch_size: int = 16,
) -> Dict[str, float]:
    import lpips
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm.auto import tqdm

    assert len(paths_a) == len(paths_b)
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()
    tf = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            ),
        ]
    )

    vals: List[float] = []
    with torch.inference_mode():
        for i in tqdm(
            range(0, len(paths_a), batch_size), desc="LPIPS"
        ):
            xa, xb = [], []
            for a, b in zip(
                paths_a[i : i + batch_size],
                paths_b[i : i + batch_size],
            ):
                xa.append(tf(Image.open(a).convert("RGB")))
                xb.append(tf(Image.open(b).convert("RGB")))
            d = loss_fn(
                torch.stack(xa).to(device), torch.stack(xb).to(device)
            )
            vals.extend(d.detach().float().cpu().reshape(-1).tolist())

    arr = np.asarray(vals, dtype=np.float64)
    out = {
        "lpips_mean": float(arr.mean()),
        "lpips_std": float(arr.std()),
        "lpips_median": float(np.median(arr)),
        "prior_preservation": float(1.0 - arr.mean()),
        "n": int(len(arr)),
    }
    del loss_fn
    free_mem()
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def write_summary_csv(
    path: Path,
    results: Dict[str, Any],
    n_eval: int,
) -> None:
    import csv

    lp = results.get("lpips") or {}
    rows = [
        {
            "Model": "SD1.4 (paper Table)",
            "FID": 13.52,
            "CLIP": 30.95,
            "LPIPS": "",
            "Prior_1_minus_LPIPS": "",
            "N": "",
        },
        {
            "Model": "DUO paper SD1.4 utility ref (nudity beta~250)",
            "FID": 13.59,
            "CLIP": 30.84,
            "LPIPS": "",
            "Prior_1_minus_LPIPS": "",
            "N": "",
        },
        {
            "Model": "BASE (this run)",
            "FID": results.get("fid_base_vs_coco", ""),
            "CLIP": results.get("clip_base", ""),
            "LPIPS": "",
            "Prior_1_minus_LPIPS": "",
            "N": n_eval,
        },
        {
            "Model": "UNLEARN / DUO (this run)",
            "FID": results.get("fid_unlearn_vs_coco", ""),
            "CLIP": results.get("clip_unlearn", ""),
            "LPIPS": lp.get("lpips_mean", ""),
            "Prior_1_minus_LPIPS": lp.get("prior_preservation", ""),
            "N": n_eval,
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Model",
                "FID",
                "CLIP",
                "LPIPS",
                "Prior_1_minus_LPIPS",
                "N",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", path)


def save_side_by_side(
    base_imgs: Sequence[Path],
    unlearn_imgs: Sequence[Path],
    captions: Sequence[str],
    out_path: Path,
    n_show: int = 6,
) -> None:
    import matplotlib
    import numpy as np
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(n_show, len(base_imgs), len(unlearn_imgs), len(captions))
    if n_show < 1:
        return
    fig, axes = plt.subplots(n_show, 2, figsize=(8, 3 * n_show))
    if n_show == 1:
        axes = np.array([axes])
    for i in range(n_show):
        ib = Image.open(base_imgs[i]).convert("RGB")
        iu = Image.open(unlearn_imgs[i]).convert("RGB")
        axes[i, 0].imshow(ib)
        axes[i, 0].set_title(
            "BASE | " + captions[i][:55], fontsize=8
        )
        axes[i, 0].axis("off")
        axes[i, 1].imshow(iu)
        axes[i, 1].set_title("UNLEARN (DUO)", fontsize=8)
        axes[i, 1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("Saved", out_path)


def zip_bundle(output_dir: Path, zip_path: Path, n_preview: int = 20) -> None:
    sample_dir = output_dir / "samples_preview"
    sample_dir.mkdir(exist_ok=True)
    base_imgs = sorted((output_dir / "images_base").glob("*.png"))
    unlearn_imgs = sorted((output_dir / "images_unlearn").glob("*.png"))
    for i in range(min(n_preview, max(len(base_imgs), len(unlearn_imgs)))):
        if i < len(base_imgs):
            shutil.copy(
                base_imgs[i], sample_dir / f"base_{i:06d}.png"
            )
        if i < len(unlearn_imgs):
            shutil.copy(
                unlearn_imgs[i], sample_dir / f"unlearn_{i:06d}.png"
            )

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        for name in (
            "metrics.json",
            "summary.csv",
            "captions_used.json",
            "side_by_side.png",
        ):
            p = output_dir / name
            if p.exists():
                z.write(p, arcname=name)
        for p in sample_dir.glob("*.png"):
            z.write(p, arcname=f"samples_preview/{p.name}")
    print("Bundle:", zip_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate DUO unlearned model utility on MS-COCO "
            "(FID / CLIP / LPIPS)."
        )
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
        help="Dir containing pytorch_lora_weights.safetensors (identity LoRA).",
    )
    p.add_argument(
        "--exp_type",
        type=str,
        default="identity",
        choices=["identity", "custom", "base_only"],
        help="identity/custom: load --lora_path; base_only: skip unlearn gen.",
    )
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/coco_eval",
    )
    p.add_argument(
        "--generate_base",
        action="store_true",
        default=True,
        help="Generate base SD images (default True).",
    )
    p.add_argument(
        "--no_generate_base",
        action="store_true",
        help="Skip base generation (use existing images_base/).",
    )
    p.add_argument(
        "--generate_unlearn",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no_generate_unlearn",
        action="store_true",
        help="Skip unlearn generation.",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--compute_fid_vs_coco", action="store_true", default=True)
    p.add_argument("--no_fid_vs_coco", action="store_true")
    p.add_argument("--compute_fid_vs_base", action="store_true", default=True)
    p.add_argument("--no_fid_vs_base", action="store_true")
    p.add_argument("--compute_clip", action="store_true", default=True)
    p.add_argument("--no_clip", action="store_true")
    p.add_argument("--compute_lpips", action="store_true", default=True)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--metric_batch", type=int, default=32)
    p.add_argument(
        "--zip_bundle",
        action="store_true",
        default=True,
        help="Write metrics_bundle.zip under output_dir parent or output_dir.",
    )
    p.add_argument("--no_zip_bundle", action="store_true")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu (default: auto)",
    )
    return p.parse_args(argv)


def _flag(default_true: bool, no_flag: bool) -> bool:
    if no_flag:
        return False
    return default_true


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    generate_base = _flag(True, args.no_generate_base)
    generate_unlearn = _flag(True, args.no_generate_unlearn)
    resume = _flag(True, args.no_resume)
    compute_fid_vs_coco = _flag(True, args.no_fid_vs_coco)
    compute_fid_vs_base = _flag(True, args.no_fid_vs_base)
    compute_clip = _flag(True, args.no_clip)
    compute_lpips = _flag(True, args.no_lpips)
    do_zip = _flag(True, args.no_zip_bundle)

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")
    if device == "cpu":
        print(
            "WARNING: No GPU — generation will be extremely slow."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_base = output_dir / "images_base"
    dir_unlearn = output_dir / "images_unlearn"
    dir_real = output_dir / "images_coco_real"
    metrics_json = output_dir / "metrics.json"
    captions_json = output_dir / "captions_used.json"

    # ---- captions ----
    records, coco_ds_ref, coco_id_to_idx = load_coco_captions(
        args.num_samples, args.seed
    )
    with open(captions_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "i": i,
                    "image_id": r["image_id"],
                    "caption": r["caption"],
                }
                for i, r in enumerate(records)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    captions = [r["caption"] for r in records]
    print("Saved", captions_json)

    # ---- generate ----
    exp_type = args.exp_type
    if exp_type == "base_only":
        generate_unlearn = False

    if generate_base:
        print("\n=== [1/2] BASE SD ===")
        pipe_base = build_sd_pipeline(
            args.pretrained_model_name_or_path,
            device,
            dtype,
            lora_path=None,
        )
        generate_images(
            pipe_base,
            captions,
            dir_base,
            args.seed,
            resume,
            "base",
            device,
            args.num_inference_steps,
            args.guidance_scale,
            args.image_size,
        )
        del pipe_base
        free_mem()
    else:
        print(
            "Skip base gen; existing:",
            len(list(dir_base.glob("*.png"))),
        )

    need_unlearn = generate_unlearn and args.lora_path is not None
    if need_unlearn:
        print("\n=== [2/2] UNLEARNED (identity LoRA) ===")
        pipe_u = build_sd_pipeline(
            args.pretrained_model_name_or_path,
            device,
            dtype,
            lora_path=args.lora_path,
        )
        generate_images(
            pipe_u,
            captions,
            dir_unlearn,
            args.seed,
            resume,
            "unlearn",
            device,
            args.num_inference_steps,
            args.guidance_scale,
            args.image_size,
        )
        del pipe_u
        free_mem()
    elif generate_unlearn:
        print(
            "WARNING: --generate_unlearn but no --lora_path. "
            "Only base metrics will be available."
        )

    # ---- real COCO for FID ----
    if compute_fid_vs_coco:
        export_real_coco(
            records,
            coco_ds_ref,
            coco_id_to_idx,
            dir_real,
            args.image_size,
            resume,
        )

    # free HF dataset if possible
    del coco_ds_ref
    free_mem()

    # ---- metrics ----
    base_imgs = sorted(dir_base.glob("*.png"))
    unlearn_imgs = sorted(dir_unlearn.glob("*.png"))
    real_imgs = sorted(dir_real.glob("*.png"))
    print(
        f"#base={len(base_imgs)} #unlearn={len(unlearn_imgs)} "
        f"#real={len(real_imgs)}"
    )

    n_eval = len(captions)
    if base_imgs:
        n_eval = min(n_eval, len(base_imgs))
    if unlearn_imgs:
        n_eval = min(n_eval, len(unlearn_imgs))
    captions_eval = captions[:n_eval]
    base_imgs = base_imgs[:n_eval] if base_imgs else base_imgs
    unlearn_imgs = (
        unlearn_imgs[:n_eval] if unlearn_imgs else unlearn_imgs
    )
    print("n_eval =", n_eval)

    results: Dict[str, Any] = {
        "config": {
            "base_model": args.pretrained_model_name_or_path,
            "lora_path": args.lora_path,
            "exp_type": exp_type,
            "num_samples_requested": args.num_samples,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": device,
        },
        "paper_reference": PAPER_REF,
        "n_eval": n_eval,
    }

    if compute_clip and base_imgs:
        print("\n[CLIP] base ...")
        results["clip_base"] = compute_clip_score(
            base_imgs, captions_eval, device, args.metric_batch
        )
        print("  CLIP base =", results["clip_base"])

    if compute_clip and unlearn_imgs:
        print("\n[CLIP] unlearn ...")
        results["clip_unlearn"] = compute_clip_score(
            unlearn_imgs, captions_eval, device, args.metric_batch
        )
        print("  CLIP unlearn =", results["clip_unlearn"])

    if compute_fid_vs_coco and real_imgs and base_imgs:
        print("\n[FID] base vs COCO real ...")
        results["fid_base_vs_coco"] = compute_fid(
            real_imgs, base_imgs, device, args.metric_batch
        )
        print("  FID base =", results["fid_base_vs_coco"])

    if compute_fid_vs_coco and real_imgs and unlearn_imgs:
        print("\n[FID] unlearn vs COCO real ...")
        results["fid_unlearn_vs_coco"] = compute_fid(
            real_imgs, unlearn_imgs, device, args.metric_batch
        )
        print("  FID unlearn =", results["fid_unlearn_vs_coco"])

    if compute_fid_vs_base and base_imgs and unlearn_imgs:
        print("\n[FID] unlearn vs base gens ...")
        results["fid_unlearn_vs_base"] = compute_fid(
            base_imgs, unlearn_imgs, device, args.metric_batch
        )
        print("  FID(unlearn, base) =", results["fid_unlearn_vs_base"])

    if compute_lpips and base_imgs and unlearn_imgs:
        n = min(len(base_imgs), len(unlearn_imgs))
        print(f"\n[LPIPS] paired n={n} ...")
        results["lpips"] = compute_lpips_paired(
            base_imgs[:n],
            unlearn_imgs[:n],
            device,
            args.image_size,
            batch_size=16,
        )
        print("  LPIPS =", results["lpips"])

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved", metrics_json)

    write_summary_csv(output_dir / "summary.csv", results, n_eval)

    if base_imgs and unlearn_imgs:
        save_side_by_side(
            base_imgs,
            unlearn_imgs,
            captions_eval,
            output_dir / "side_by_side.png",
        )

    if do_zip:
        zip_path = output_dir / "duo_coco_metrics_bundle.zip"
        zip_bundle(output_dir, zip_path)

    print("\n===== SUMMARY =====")
    print(json.dumps(results, indent=2))
    print(
        "\nHow to read:\n"
        "- FID unlearn ~ FID base ~ 13-14 (large N) => utility kept\n"
        "- CLIP unlearn not much worse than base => text alignment OK\n"
        "- Low LPIPS / high (1-LPIPS) => prior preservation good\n"
        "- Absolute numbers may differ from paper if N<<30k, steps, "
        "CLIP backbone, or FID real-set differ\n"
        "- Paper table: NUM_SAMPLES=30000, consider --num_inference_steps 50"
    )
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
