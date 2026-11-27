#!/usr/bin/env python3
"""Headless CLI to build and audit the LOC By the People (LCCN 2020446971) challenge panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eahtr.loc_challenge import build_loc_challenge_manifest


def _load_revcity_splits(manifests_dir: Path) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test_public"):
        path = manifests_dir / f"{split_name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"RevCity split manifest not found: {path}")
        splits[split_name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LOC By the People Public Domain evaluation challenge panel.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/loc_btp"), help="Output dir for pages/, manifests/, and audit JSON")
    parser.add_argument("--zip-path", type=Path, default=None, help="Optional existing LOC 2020446971 zip archive")
    parser.add_argument("--max-pages", type=int, default=150, help="Maximum admitted challenge pages (0 for all rows)")
    parser.add_argument("--path-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--image-delay-s", type=float, default=0.4, help="Polite delay between LOC image requests")
    parser.add_argument("--revcity-manifests-dir", type=Path, default=None, help="Optional RevCity manifests/ dir for cross-corpus disjointness audit")
    args = parser.parse_args()

    revcity_splits = _load_revcity_splits(args.revcity_manifests_dir) if args.revcity_manifests_dir else None
    max_pages = None if args.max_pages <= 0 else args.max_pages
    try:
        res = build_loc_challenge_manifest(
            args.out_dir,
            max_pages=max_pages,
            path_mode=args.path_mode,
            zip_path=args.zip_path,
            image_delay_s=args.image_delay_s,
            revcity_splits=revcity_splits,
            write_outputs=True,
        )
    except Exception as exc:
        print(f"[ERROR] LOC challenge build failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(res["audit"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
