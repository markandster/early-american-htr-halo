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
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="HF repo ID or staged model directory for exact token counts")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=1600)
    parser.add_argument("--vision-tokens", type=int, default=1120)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    count_fn = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    train_rows = _load_jsonl(args.train_jsonl)
    test_rows = _load_jsonl(args.test_jsonl)

    if args.mode == "sft":
        res = export_halo_vlm_sft_dataset(
            train_rows, test_rows, args.out_dir, count_fn,
            max_length=args.max_length, vision_tokens=args.vision_tokens,
        )
    else:
        res = export_halo_vlm_preference_dataset(
            train_rows, test_rows, args.out_dir, count_fn,
            max_length=args.max_length, max_prompt_length=args.max_prompt_length, vision_tokens=args.vision_tokens,
        )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
