#!/usr/bin/env python3
"""Export full-scale Halo SFT or VLM SMPO/DPO Arrow DatasetDicts with post-expansion filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eahtr.halo_export import export_halo_vlm_preference_dataset, export_halo_vlm_sft_dataset


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Halo SFT or VLM SMPO/DPO Arrow DatasetDict.")
    parser.add_argument("--mode", choices=("sft", "preference"), required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", "--test-jsonl", dest="val_jsonl", type=Path, required=True, help="Dev validation JSONL (mapped to Halo internal test split)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="HF repo ID or staged model directory for exact token counts")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=1600)
    parser.add_argument("--vision-tokens", type=int, default=1120)
    parser.add_argument("--measure-vision-tokens", action="store_true", help="Measure realized image tokens from first sample image using AutoProcessor")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    count_fn = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    train_rows = _load_jsonl(args.train_jsonl)
    val_rows = _load_jsonl(args.val_jsonl)

    vision_tokens = args.vision_tokens
    if args.measure_vision_tokens and train_rows:
        from PIL import Image
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(args.tokenizer)
        first = train_rows[0]
        img_path = first.get("image_path") or (first.get("images") or [None])[0]
        if img_path and Path(str(img_path)).is_file():
            with Image.open(str(img_path)) as im:
                rendered = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Transcribe."}]}], tokenize=False, add_generation_prompt=True)
                enc = proc(text=[rendered], images=[im.convert("RGB")], return_tensors="pt")
            img_tid = getattr(tok, "image_token_id", None) or tok.get_vocab().get("<|image|>") or tok.get_vocab().get("<image_soft_token>")
            if img_tid is not None:
                vision_tokens = sum(1 for t in enc["input_ids"][0].tolist() if int(t) == int(img_tid))

    if args.mode == "sft":
        res = export_halo_vlm_sft_dataset(
            train_rows, val_rows, args.out_dir, count_fn,
            max_length=args.max_length, vision_tokens=vision_tokens,
        )
    else:
        res = export_halo_vlm_preference_dataset(
            train_rows, val_rows, args.out_dir, count_fn,
            max_length=args.max_length, max_prompt_length=args.max_prompt_length, vision_tokens=vision_tokens,
        )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
