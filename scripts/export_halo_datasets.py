#!/usr/bin/env python3
"""Export full-scale Halo SFT or VLM SMPO Arrow DatasetDicts with post-expansion filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eahtr.halo_export import export_halo_vlm_preference_dataset, export_halo_vlm_sft_dataset
from eahtr.revcity_build import resolve_manifest_image_path
from eahtr.vision_budget import measure_max_vision_tokens, parse_cell_size


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _resolve_row_paths(rows: list[dict], manifest_path: Path, image_root: Path | None) -> list[dict]:
    resolved_rows: list[dict] = []
    for row in rows:
        item = dict(row)
        if item.get("image_path"):
            item["image_path"] = str(resolve_manifest_image_path(str(item["image_path"]), manifest_path, image_root))
        if item.get("images"):
            item["images"] = [
                str(resolve_manifest_image_path(str(p), manifest_path, image_root))
                for p in item["images"]
            ]
        resolved_rows.append(item)
    return resolved_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Halo SFT or VLM SMPO Arrow DatasetDict.")
    parser.add_argument("--mode", choices=("sft", "preference"), required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", "--test-jsonl", dest="val_jsonl", type=Path, required=True, help="Dev validation JSONL (mapped to Halo internal test split)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="HF repo ID or staged model directory for exact token counts")
    parser.add_argument("--revision", default=None, help="Optional pinned HF revision for --tokenizer")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional directory overriding manifest image root")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=1600)
    parser.add_argument("--vision-tokens", type=int, default=1120)
    parser.add_argument("--max-image-tokens", type=int, default=None, help="Resize images to exact grid-cell multiples within this image-token budget")
    parser.add_argument("--cell-size", default=None, help="Optional pixel grid cell size 'W,H' or 'N' (auto-detected from staged processor_config.json when omitted)")
    parser.add_argument("--image-format", choices=("PNG", "JPEG"), default="PNG")
    parser.add_argument("--measure-vision-tokens", action="store_true", help="Measure realized image tokens across up to 8 sample images using AutoProcessor")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok_kw = {"revision": args.revision} if args.revision else {}
    tok = AutoTokenizer.from_pretrained(args.tokenizer, **tok_kw)
    count_fn = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    train_rows = _resolve_row_paths(_load_jsonl(args.train_jsonl), args.train_jsonl, args.image_root)
    val_rows = _resolve_row_paths(_load_jsonl(args.val_jsonl), args.val_jsonl, args.image_root)
    cell_size = parse_cell_size(args.cell_size, args.tokenizer)

    vision_tokens = args.vision_tokens
    if args.measure_vision_tokens and train_rows:
        sample_paths = [
            Path(str(r.get("image_path") or r["images"][0]))
            for r in train_rows[:8]
            if r.get("image_path") or r.get("images")
        ]
        if sample_paths:
            measured = measure_max_vision_tokens(
                sample_paths,
                args.tokenizer,
                max_image_tokens=args.max_image_tokens,
                cell_size=cell_size,
                revision=args.revision,
            )
            vision_tokens = max(vision_tokens, int(measured["max_image_tokens"]))

    if args.mode == "sft":
        res = export_halo_vlm_sft_dataset(
            train_rows, val_rows, args.out_dir, count_fn,
            max_length=args.max_length,
            vision_tokens=vision_tokens,
            max_image_tokens=args.max_image_tokens,
            cell_size=cell_size,
            image_format=args.image_format,
        )
    else:
        res = export_halo_vlm_preference_dataset(
            train_rows, val_rows, args.out_dir, count_fn,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            vision_tokens=vision_tokens,
            max_image_tokens=args.max_image_tokens,
            cell_size=cell_size,
            image_format=args.image_format,
        )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
