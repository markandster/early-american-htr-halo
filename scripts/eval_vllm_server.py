#!/usr/bin/env python3
"""Evaluate a multimodal HTR model or exported adapter served behind an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from eahtr.candidates import DIPLOMATIC_PROMPT
from eahtr.eval_harness import (
    load_baseline_cer_map,
    load_eval_manifest,
    score_eval_predictions,
    write_json_atomic,
)
from eahtr.vision_budget import encode_image_for_vlm, parse_cell_size
from eahtr.vllm_client import VLLMClient, VLLMClientError, validate_vllm_engine_adapter_dir


def _count_expected_prompt_text_tokens(tokenizer_ref: str, *, revision: str | None = None) -> int:
    from transformers import AutoTokenizer

    tok_kw = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(tokenizer_ref, **tok_kw)
    sample = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}]
    try:
        encoded = tok.apply_chat_template(
            sample, tokenize=True, add_generation_prompt=True, reasoning_effort=0.0
        )
    except TypeError:
        encoded = tok.apply_chat_template(sample, tokenize=True, add_generation_prompt=True)

    raw_ids = encoded["input_ids"] if (isinstance(encoded, Mapping) or hasattr(encoded, "keys")) else encoded
    ids = raw_ids.tolist() if hasattr(raw_ids, "tolist") else list(raw_ids)
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    img_tids: set[int] = set()
    direct_tid = getattr(tok, "image_token_id", None)
    if isinstance(direct_tid, int) and direct_tid >= 0:
        img_tids.add(direct_tid)
    if hasattr(tok, "convert_tokens_to_ids"):
        unk_id = getattr(tok, "unk_token_id", None)
        for tok_str in ("<|unused_200054|>", "<|image|>", "<image_soft_token>"):
            cid = tok.convert_tokens_to_ids(tok_str)
            if isinstance(cid, int) and cid >= 0 and (unk_id is None or cid != int(unk_id)):
                img_tids.add(cid)
    return sum(1 for t in ids if int(t) not in img_tids)


def _verify_first_request_prompt_tokens(
    usage: dict[str, Any],
    *,
    img_tokens: int,
    expected_text_tokens: int | None,
    tolerance: int,
) -> None:
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens < img_tokens:
        raise RuntimeError(
            f"First-request vision parity failed: usage.prompt_tokens={prompt_tokens!r} < image_tokens={img_tokens}"
        )
    if expected_text_tokens is not None:
        expected_total = expected_text_tokens + img_tokens
        if abs(prompt_tokens - expected_total) > tolerance:
            raise RuntimeError(
                f"First-request prompt_tokens parity drift: server reported {prompt_tokens} != "
                f"expected {expected_total} (+/- {tolerance}; text={expected_text_tokens}, img={img_tokens})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VLM over OpenAI-compatible vLLM chat completions API.")
    parser.add_argument("--base-url", required=True, help="vLLM server base URL (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--model", required=True, help="Served model or LoRA adapter ID registered with vLLM")
    parser.add_argument("--eval-jsonl", type=Path, required=True, help="Evaluation manifest JSONL (val.jsonl or frozen test_public.jsonl)")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None, help="Optional directory overriding manifest image root")
    parser.add_argument("--allow-frozen-test", action="store_true")
    parser.add_argument("--baseline-predictions", type=Path, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--swap-images", action="store_true")
    parser.add_argument("--dual-convention", action="store_true")
    parser.add_argument("--max-image-tokens", type=int, default=2000)
    parser.add_argument("--cell-size", default=None)
    parser.add_argument("--image-format", choices=("PNG", "JPEG"), default="PNG")
    parser.add_argument("--engine-adapter-dir", type=Path, default=None, help="Optional exported Inkling engine adapter to validate before serving")
    parser.add_argument("--tokenizer", default=None, help="Optional HF repo/dir for strict first-request prompt_tokens parity")
    parser.add_argument("--revision", default=None, help="Optional pinned HF revision for --tokenizer")
    parser.add_argument("--prompt-token-tolerance", type=int, default=1)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--allow-insecure-remote", action="store_true")
    args = parser.parse_args()

    try:
        rows = load_eval_manifest(
            args.eval_jsonl,
            allow_frozen_test=args.allow_frozen_test,
            max_pages=args.max_pages,
            image_root=args.image_root,
        )
        base_map = None
        if args.baseline_predictions is not None:
            eval_ids = [str(r["record_id"]).strip() for r in rows]
            base_map = load_baseline_cer_map(args.baseline_predictions, eval_ids)
        cell_size = parse_cell_size(args.cell_size, args.tokenizer)
        if args.engine_adapter_dir is not None:
            validate_vllm_engine_adapter_dir(args.engine_adapter_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    api_key = os.environ.get(args.api_key_env) or None
    client = VLLMClient(
        args.base_url,
        model=args.model,
        api_key=api_key,
        timeout_s=args.timeout_s,
        allow_insecure_remote=args.allow_insecure_remote,
    )
    expected_text_tokens = (
        _count_expected_prompt_text_tokens(args.tokenizer, revision=args.revision) if args.tokenizer else None
    )

    from PIL import Image

    mime_type = "image/jpeg" if args.image_format == "JPEG" else "image/png"
    raw_predictions: list[dict[str, Any]] = []
    n_rows = len(rows)
    for idx, row in enumerate(rows):
        img_path = rows[(idx + 1) % n_rows]["image_path"] if (args.swap_images and n_rows > 1) else row["image_path"]
        raw_im = Image.open(img_path)
        try:
            payload_bytes, _, _, img_tokens = encode_image_for_vlm(
                raw_im,
                max_image_tokens=args.max_image_tokens,
                cell_w=cell_size[0],
                cell_h=cell_size[1],
                image_format=args.image_format,
            )
        finally:
            if hasattr(raw_im, "close"):
                raw_im.close()

        resp = client.transcribe_page(
            payload_bytes,
            mime_type=mime_type,
            prompt_text=DIPLOMATIC_PROMPT,
            max_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            reasoning_effort=0.0,
        )
        if idx == 0:
            _verify_first_request_prompt_tokens(
                resp["usage"],
                img_tokens=img_tokens,
                expected_text_tokens=expected_text_tokens,
                tolerance=args.prompt_token_tolerance,
            )
        raw_predictions.append(resp)

    scores, preds = score_eval_predictions(
        rows,
        raw_predictions,
        swap_images=args.swap_images,
        dual_convention=args.dual_convention,
        baseline_cer_map=base_map,
    )
    write_json_atomic(args.out_json, {"scores": scores, "predictions": preds})
    print(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VLLMClientError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
