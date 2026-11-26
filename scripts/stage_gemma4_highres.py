#!/usr/bin/env python3
"""Stage a zero-copy local Gemma 4 HF directory with 1120 vision soft tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eahtr.halo_export import verify_exported_vlm_dir
from eahtr.halo_smoke import prepare_gemma4_highres_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a local Gemma 4 high-res (1120 tokens) model directory.")
    parser.add_argument("--model", required=True, help="HF repo ID (e.g. google/gemma-4-12B-it) or local dir")
    parser.add_argument("--revision", default=None, help="Optional pinned HF commit SHA (e.g. 43c9d014b231)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output staged directory path")
    parser.add_argument("--max-soft-tokens", type=int, default=1120, help="Target Gemma 4 vision token budget")
    parser.add_argument("--chat-template", type=Path, default=None, help="Optional Jinja chat template override")
    args = parser.parse_args()

    src = Path(args.model)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(
            snapshot_download(
                args.model,
                revision=args.revision,
                allow_patterns=["*.json", "*.jinja", "*.model", "*.safetensors*", "*.tiktoken"],
            )
        )

    staged = prepare_gemma4_highres_dir(
        src,
        args.out_dir,
        max_soft_tokens=args.max_soft_tokens,
        chat_template_path=args.chat_template,
    )
    verified = verify_exported_vlm_dir(
        staged,
        expected_soft_tokens=args.max_soft_tokens,
        check_gemma4_resolution=True,
        check_nothink_template=args.chat_template is not None,
    )
    print(json.dumps(verified, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
