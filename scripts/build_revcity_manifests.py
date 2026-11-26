#!/usr/bin/env python3
"""Headless CLI to verify, clean, split, and audit Revolutionary City HTR manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from eahtr.candidates import REVCITY_ZENODO_BYTES, REVCITY_ZENODO_MD5, REVCITY_ZENODO_URL
from eahtr.revcity_build import build_revcity_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic document-disjoint RevCity HTR manifests.")
    parser.add_argument("--zip-path", type=Path, default=Path("data/revcity/revcity-htr.zip"), help="Path to revcity-htr.zip")
    parser.add_argument("--out-dir", type=Path, default=Path("data/revcity"), help="Root output directory for pages/ and manifests/")
    parser.add_argument("--download", action="store_true", help="Download revcity-htr.zip from Zenodo if absent or invalid")
    parser.add_argument("--path-mode", choices=("relative", "absolute"), default="relative", help="Manifest image_path style")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional stratified smoke page cap")
    parser.add_argument("--creator-map", type=Path, default=None, help="Optional JSON mapping stem/safe_id to creator ID")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--min-eval-pages", type=int, default=8)
    parser.add_argument("--skip-md5", action="store_true", help="Skip Zenodo byte/MD5 verification (custom archives only)")
    parser.add_argument("--export-halo-dir", type=Path, default=None, help="Optional SFT Arrow output dir to export immediately")
    parser.add_argument("--tokenizer", default=None, help="Staged Gemma 4/tokenizer path required with --export-halo-dir")
    args = parser.parse_args()

    if args.export_halo_dir is not None and not args.tokenizer:
        print("[ERROR] --tokenizer is required when --export-halo-dir is specified.", file=sys.stderr)
        return 2

    creator_map = None
    if args.creator_map is not None:
        if not args.creator_map.is_file():
            print(f"[ERROR] Creator map JSON not found: {args.creator_map}", file=sys.stderr)
            return 2
        creator_map = json.loads(args.creator_map.read_text(encoding="utf-8"))

    manifest_dir = args.out_dir / "manifests"
    pages_dir = args.out_dir / "pages"
    try:
        summary = build_revcity_manifests(
            args.zip_path,
            pages_dir,
            manifest_dir,
            seed=args.seed,
            max_pages=args.max_pages,
            creator_map=creator_map,
            path_mode=args.path_mode,
            download_url=REVCITY_ZENODO_URL if args.download else None,
            expected_bytes=None if args.skip_md5 else REVCITY_ZENODO_BYTES,
            expected_md5=None if args.skip_md5 else REVCITY_ZENODO_MD5,
            min_eval_pages=args.min_eval_pages,
        )
    except Exception as exc:
        print(f"[ERROR] RevCity manifest build failed: {exc}", file=sys.stderr)
        return 2

    if args.export_halo_dir is not None:
        export_script = Path(__file__).resolve().parent / "export_halo_datasets.py"
        subprocess.run(
            [
                sys.executable,
                str(export_script),
                "--mode",
                "sft",
                "--train-jsonl",
                str(manifest_dir / "train.jsonl"),
                "--val-jsonl",
                str(manifest_dir / "val.jsonl"),
                "--out-dir",
                str(args.export_halo_dir),
                "--tokenizer",
                str(args.tokenizer),
            ],
            check=True,
        )

    printable = {
        "out_dir": str(args.out_dir),
        "manifest_dir": str(manifest_dir),
        "path_mode": summary["path_mode"],
        "splits": summary["splits"],
        "hard_case_panel_pages": summary["hard_case_panel_pages"],
        "hard_case_held_out_pages": summary["hard_case_held_out_pages"],
        "dhash_review_candidates": len(summary["dhash_review_candidates"]),
    }
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
