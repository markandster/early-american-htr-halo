#!/usr/bin/env python3
"""Merge a Halo PEFT adapter (optional) and evaluate a single checkpoint on val or frozen test_public."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from eahtr.candidates import DIPLOMATIC_PROMPT
from eahtr.eval_harness import (
    load_baseline_cer_map,
    load_eval_manifest,
    score_eval_predictions,
    write_json_atomic,
)
from eahtr.halo_export import verify_exported_vlm_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Halo adapter (optional) and evaluate VLM on JSONL panel.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Merged HF checkpoint directory")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Optional Halo PEFT adapter dir to merge into --model-dir first")
    parser.add_argument("--halo-repo", type=Path, default=Path("/tmp/halo"), help="Halo checkout containing merge_peft_adapters.py")
    parser.add_argument("--eval-jsonl", type=Path, required=True, help="Evaluation manifest JSONL (val.jsonl or frozen test_public.jsonl)")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional directory overriding manifest image root")
    parser.add_argument("--allow-frozen-test", action="store_true", help="Required guard to evaluate test_public manifests")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, default=None, help="Optional baseline JSON for paired cluster bootstrap CIs")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Fixed inference-time decode token cap")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--swap-images", action="store_true", help="Permute image paths across pages to verify vision grounding")
    parser.add_argument("--dual-convention", action="store_true", help="Also report raw_strict_cer_pct against row['raw_text']")
    parser.add_argument("--model-family", choices=("gemma4", "inkling2"), default="gemma4")
    args = parser.parse_args()

    try:
        rows = load_eval_manifest(
            args.eval_jsonl,
            allow_frozen_test=args.allow_frozen_test,
            max_pages=args.max_pages,
            image_root=args.image_root,
        )
        base_map = None
        if args.baseline_predictions is not None:
            eval_ids = [str(r["record_id"]).strip() for r in rows]
            base_map = load_baseline_cer_map(args.baseline_predictions, eval_ids)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.adapter_dir is not None:
        merge_script = args.halo_repo / "scripts" / "after_training" / "merge_peft_adapters.py"
        subprocess.run(
            [sys.executable, str(merge_script), "--adapter_dir", str(args.adapter_dir), "--output_dir", str(args.model_dir)],
            check=True,
        )

    verify_exported_vlm_dir(args.model_dir, model_family=args.model_family, reference_dir=args.adapter_dir)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa" if args.model_family == "gemma4" else "eager",
    )
    model.eval()

    raw_predictions: list[dict] = []
    n_rows = len(rows)
    cap = int(args.max_new_tokens)
    for idx, row in enumerate(rows):
        img_path = rows[(idx + 1) % n_rows]["image_path"] if (args.swap_images and n_rows > 1) else row["image_path"]
        raw_im = Image.open(img_path)
        rgb_im = None
        try:
            rgb_im = raw_im.convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[prompt], images=[rgb_im], return_tensors="pt").to(model.device)
        finally:
            if rgb_im is not None and rgb_im is not raw_im and hasattr(rgb_im, "close"):
                rgb_im.close()
            if hasattr(raw_im, "close"):
                raw_im.close()

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cap, do_sample=False, repetition_penalty=args.repetition_penalty)
        gen_ids = out[:, inputs["input_ids"].shape[1] :]
        fr = "length" if int(gen_ids.shape[1]) >= cap else "stop"
        raw_gen = proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
        gen = proc.batch_decode(gen_ids, skip_special_tokens=True)[0]
        raw_predictions.append({"text": gen, "raw_text": raw_gen, "finish_reason": fr})

    scores, preds = score_eval_predictions(
        rows,
        raw_predictions,
        swap_images=args.swap_images,
        dual_convention=args.dual_convention,
        baseline_cer_map=base_map,
    )
    write_json_atomic(args.out_json, {"scores": scores, "predictions": preds})
    print(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
