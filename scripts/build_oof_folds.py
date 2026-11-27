#!/usr/bin/env python3
"""Headless CLI to generate K balanced group-disjoint OOF Stage-2 manifests from train.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eahtr.oof_folds import build_oof_fold_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Build balanced group-disjoint out-of-fold manifests from train.jsonl.")
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for oof_fold{k}_{train,holdout}.jsonl")
    parser.add_argument("--num-folds", type=int, default=4, help="Number of OOF folds (>= 2)")
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()

    try:
        summary = build_oof_fold_manifests(
            args.train_jsonl,
            args.out_dir,
            num_folds=args.num_folds,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"[ERROR] OOF fold generation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
