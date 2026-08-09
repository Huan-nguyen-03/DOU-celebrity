#!/usr/bin/env python3
"""
Full identity-unlearning evaluation (DUO-paper structure, identity domain).

Runs:
  (1) Identity face metrics + red-team suites
        E1, E2, E4, E_RAB_identity, E_Sneaky_identity
        → ISR / IER via ArcFace
  (2) COCO utility (optional, default on)
        FID / CLIP / LPIPS

Usage:
  python -m eval.run_full_identity_eval \\
      --lora_path .../Identity_Obama/checkpoint-500 \\
      --identity_name "Barack Obama" \\
      --gallery_dir /path/to/obama_photos \\
      --output_dir outputs/full_identity_eval \\
      --num_seeds 5 \\
      --coco_num_samples 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full identity eval: ArcFace suites (+redteam) + COCO utility"
    )
    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    p.add_argument("--lora_path", type=str, required=True)
    p.add_argument("--identity_name", type=str, default="Barack Obama")
    p.add_argument("--identity_short", type=str, default="Obama")
    p.add_argument("--gallery_dir", type=str, default=None)
    p.add_argument("--gallery_synthetic_n", type=int, default=8)
    p.add_argument("--eval_models", type=str, default="base,unlearn")
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--seed_base", type=int, default=0)
    p.add_argument("--max_prompts_per_suite", type=int, default=None)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--tau", type=float, default=0.35)
    p.add_argument("--no_auto_calibrate_tau", action="store_true")
    p.add_argument("--no_redteam", action="store_true")
    p.add_argument("--no_redteam_clip_search", action="store_true")
    p.add_argument("--n_rab", type=int, default=24)
    p.add_argument("--n_sneaky", type=int, default=32)
    p.add_argument("--rab_prompt_file", type=str, default=None)
    p.add_argument("--sneaky_prompt_file", type=str, default=None)
    p.add_argument(
        "--output_dir", type=str, default="./outputs/full_identity_eval"
    )
    p.add_argument("--device", type=str, default=None)

    # which stages
    p.add_argument(
        "--skip_identity",
        action="store_true",
        help="Skip ArcFace / red-team stage",
    )
    p.add_argument(
        "--skip_coco",
        action="store_true",
        help="Skip COCO FID/CLIP/LPIPS stage",
    )
    p.add_argument("--coco_num_samples", type=int, default=1000)
    p.add_argument("--coco_steps", type=int, default=None)
    p.add_argument("--no_save_images", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"stages": {}}

    # ----- Stage 1: identity + redteam -----
    if not args.skip_identity:
        from eval.identity_metrics import parse_args as id_parse
        from eval.identity_metrics import run as id_run

        id_out = out / "identity"
        id_argv: List[str] = [
            "--pretrained_model_name_or_path",
            args.pretrained_model_name_or_path,
            "--lora_path",
            args.lora_path,
            "--identity_name",
            args.identity_name,
            "--identity_short",
            args.identity_short,
            "--eval_models",
            args.eval_models,
            "--num_seeds",
            str(args.num_seeds),
            "--seed_base",
            str(args.seed_base),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--guidance_scale",
            str(args.guidance_scale),
            "--image_size",
            str(args.image_size),
            "--tau",
            str(args.tau),
            "--output_dir",
            str(id_out),
            "--n_rab",
            str(args.n_rab),
            "--n_sneaky",
            str(args.n_sneaky),
            "--gallery_synthetic_n",
            str(args.gallery_synthetic_n),
        ]
        if args.gallery_dir:
            id_argv += ["--gallery_dir", args.gallery_dir]
        if args.max_prompts_per_suite is not None:
            id_argv += [
                "--max_prompts_per_suite",
                str(args.max_prompts_per_suite),
            ]
        if args.no_auto_calibrate_tau:
            id_argv.append("--no_auto_calibrate_tau")
        if args.no_redteam:
            id_argv.append("--no_redteam")
        if args.no_redteam_clip_search:
            id_argv.append("--no_redteam_clip_search")
        if args.rab_prompt_file:
            id_argv += ["--rab_prompt_file", args.rab_prompt_file]
        if args.sneaky_prompt_file:
            id_argv += ["--sneaky_prompt_file", args.sneaky_prompt_file]
        if args.no_save_images:
            id_argv.append("--no_save_images")
        if args.device:
            id_argv += ["--device", args.device]

        print("\n" + "=" * 60)
        print("STAGE 1/2 — Identity + red-team (ArcFace ISR/IER)")
        print("=" * 60)
        id_args = id_parse(id_argv)
        report["stages"]["identity"] = id_run(id_args)
    else:
        print("Skip identity stage")

    # ----- Stage 2: COCO utility -----
    if not args.skip_coco:
        from eval.coco_metrics import parse_args as coco_parse
        from eval.coco_metrics import run as coco_run

        coco_out = out / "coco"
        coco_steps = args.coco_steps or args.num_inference_steps
        coco_argv = [
            "--pretrained_model_name_or_path",
            args.pretrained_model_name_or_path,
            "--lora_path",
            args.lora_path,
            "--exp_type",
            "identity",
            "--num_samples",
            str(args.coco_num_samples),
            "--num_inference_steps",
            str(coco_steps),
            "--guidance_scale",
            str(args.guidance_scale),
            "--image_size",
            str(args.image_size),
            "--seed",
            str(args.seed_base),
            "--output_dir",
            str(coco_out),
        ]
        if args.device:
            coco_argv += ["--device", args.device]

        print("\n" + "=" * 60)
        print("STAGE 2/2 — COCO utility (FID / CLIP / LPIPS)")
        print("=" * 60)
        coco_args = coco_parse(coco_argv)
        report["stages"]["coco"] = coco_run(coco_args)
    else:
        print("Skip COCO stage")

    # ----- Combined headline -----
    headline: Dict[str, Any] = {}
    id_sum = (report.get("stages") or {}).get("identity", {}).get("summary")
    if id_sum:
        for row in id_sum:
            key = f"{row['model']}__{row['suite']}"
            headline[key] = {
                "ISR": row["ISR"],
                "IER": row["IER"],
                "MeanSim": row["MeanSim"],
            }
    coco = (report.get("stages") or {}).get("coco") or {}
    if coco:
        headline["coco"] = {
            "fid_base_vs_coco": coco.get("fid_base_vs_coco"),
            "fid_unlearn_vs_coco": coco.get("fid_unlearn_vs_coco"),
            "clip_base": coco.get("clip_base"),
            "clip_unlearn": coco.get("clip_unlearn"),
            "lpips": coco.get("lpips"),
        }

    report["headline"] = headline
    report_path = out / "full_eval_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("\nSaved full report:", report_path)

    print("\n===== FULL IDENTITY EVAL HEADLINE =====")
    print(json.dumps(headline, indent=2, default=str))
    print(
        "\nHow to read (same structure as DUO paper, identity domain):\n"
        "  • E1/E2/E_RAB/E_Sneaky: unlearn IER ↑ / ISR ↓ vs base\n"
        "  • E4_related: do not collapse unrelated people\n"
        "  • COCO: FID/CLIP of unlearn ≈ base; LPIPS not huge\n"
    )
    return report


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
