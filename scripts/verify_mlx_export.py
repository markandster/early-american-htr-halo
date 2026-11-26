#!/usr/bin/env python3
"""Verify merged HF or converted MLX VLM output directories retain 1120 tokens and non-thinking template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eahtr.halo_export import verify_exported_vlm_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify merged HF or converted MLX VLM checkpoint directory.")
    parser.add_argument("model_dir", type=Path, help="Path to merged_hf or mlx_out directory")
    parser.add_argument("--expected-soft-tokens", type=int, default=1120)
    parser.add_argument("--skip-gemma4-resolution", action="store_true")
    parser.add_argument("--skip-template-check", action="store_true")
    args = parser.parse_args()

    try:
        res = verify_exported_vlm_dir(
            args.model_dir,
            expected_soft_tokens=args.expected_soft_tokens,
            check_gemma4_resolution=not args.skip_gemma4_resolution,
            check_nothink_template=not args.skip_template_check,
        )
    except Exception as exc:
        print(f"[ERROR] Export verification failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
