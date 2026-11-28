#!/usr/bin/env python3
"""Merge multi-fold OOF SMPO preference JSONLs with eval-leakage auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eahtr.eval_harness import write_json_atomic
from eahtr.oof_preferences import merge_and_audit_oof_preferences


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge K-fold OOF preference shards and audit against eval manifests.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--eval-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--min-reward-margin", type=float, default=0.05)
    parser.add_argument("--require-distinct-folds", action="store_true")
    parser.add_argument("--out-summary", type=Path, default=None)
    args = parser.parse_args()

    try:
        summary = merge_and_audit_oof_preferences(
            args.inputs,
            args.out_jsonl,
            eval_manifests=args.eval_manifests,
            min_reward_margin=args.min_reward_margin,
            require_distinct_folds=args.require_distinct_folds,
        )
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"[ERROR] OOF preference merge failed: {exc}", file=sys.stderr)
        return 2

    if args.out_summary is not None:
        write_json_atomic(args.out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
