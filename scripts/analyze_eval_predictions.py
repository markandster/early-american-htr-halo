#!/usr/bin/env python3
"""Diagnostic CLI for per-page HTR error analysis, line-break drift, and loop/cap buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eahtr.eval_analysis import analyze_predictions, load_predictions_json


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze per-page HTR predictions and failure modes.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions_*.json or report JSON")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline predictions JSON for paired delta")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest for two-page opening breakdown")
    parser.add_argument("--top-k", type=int, default=5, help="Number of worst-CER pages to include")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output file to write JSON analysis")
    args = parser.parse_args()

    preds = load_predictions_json(args.predictions)
    baseline = load_predictions_json(args.baseline) if args.baseline else None
    manifest = _load_jsonl(args.manifest) if args.manifest else None
    report = analyze_predictions(preds, baseline_predictions=baseline, manifest_rows=manifest, top_k=args.top_k)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
