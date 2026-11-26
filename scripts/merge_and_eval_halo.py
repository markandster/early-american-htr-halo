#!/usr/bin/env python3
"""Merge a Halo PEFT adapter (optional) and evaluate a single checkpoint on val or frozen test_public."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from eahtr.candidates import DIPLOMATIC_PROMPT
from eahtr.halo_export import verify_exported_vlm_dir
from eahtr.metrics import normalize_preserve_newlines, paired_bootstrap_ci, score_line_breaks, score_pair
from eahtr.rewards import score_transcription


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Halo adapter (optional) and evaluate VLM on JSONL panel.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Merged HF checkpoint directory")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Optional Halo PEFT adapter dir to merge into --model-dir first")
    parser.add_argument("--halo-repo", type=Path, default=Path("/tmp/halo"), help="Halo checkout containing merge_peft_adapters.py")
    parser.add_argument("--eval-jsonl", type=Path, required=True, help="Evaluation manifest JSONL (val.jsonl or frozen test_public.jsonl)")
    parser.add_argument("--allow-frozen-test", action="store_true", help="Required guard to evaluate test_public manifests")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, default=None, help="Optional baseline JSON for paired cluster bootstrap CIs")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Fixed inference-time decode token cap")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--swap-images", action="store_true", help="Permute image paths across pages to verify vision grounding")
    parser.add_argument("--dual-convention", action="store_true", help="Also report raw_strict_cer_pct against row['raw_text']")
    parser.add_argument("--model-family", choices=("gemma4", "inkling2"), default="gemma4")
    args = parser.parse_args()

    rows = _load_jsonl(args.eval_jsonl)
    if args.max_pages:
        rows = rows[: args.max_pages]
    if not rows:
        print(f"Empty evaluation manifest: {args.eval_jsonl}", file=sys.stderr)
        return 2
    has_test_split = "test" in args.eval_jsonl.name.lower() or any("test" in str(r.get("split", "")).lower() for r in rows)
    if has_test_split and not args.allow_frozen_test:
        print("Refusing test manifest without --allow-frozen-test during dev. Use val.jsonl.", file=sys.stderr)
        return 2
    eval_ids_list = [str(r.get("record_id") or "").strip() for r in rows]
    if any(not rid for rid in eval_ids_list) or len(eval_ids_list) != len(set(eval_ids_list)):
        print(f"Missing or duplicate record_id entries found in {args.eval_jsonl}", file=sys.stderr)
        return 2

    base_map: dict[str, float] | None = None
    if args.baseline_predictions is not None:
        if not args.baseline_predictions.is_file():
            print(f"Baseline predictions file not found: {args.baseline_predictions}", file=sys.stderr)
            return 2
        base_data = json.loads(args.baseline_predictions.read_text(encoding="utf-8"))
        base_rows = base_data if isinstance(base_data, list) else base_data.get("predictions", [])
        base_ids_list = [str(r.get("record_id") or "").strip() for r in base_rows]
        if any(not rid for rid in base_ids_list) or len(base_ids_list) != len(set(base_ids_list)):
            print(f"Missing or duplicate record_id entries found in {args.baseline_predictions}", file=sys.stderr)
            return 2
        if set(base_ids_list) != set(eval_ids_list):
            print(
                f"Baseline predictions must match eval record_ids 1-to-1 (base={len(base_ids_list)}, eval={len(eval_ids_list)})",
                file=sys.stderr,
            )
            return 2
        try:
            base_map = {}
            for r in base_rows:
                val = float(r["strict_cer"])
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite strict_cer={val!r}")
                base_map[str(r["record_id"]).strip()] = val
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Invalid strict_cer entry in {args.baseline_predictions}: {exc}", file=sys.stderr)
            return 2

    if args.adapter_dir is not None:
        merge_script = args.halo_repo / "scripts" / "after_training" / "merge_peft_adapters.py"
        subprocess.run(
            [sys.executable, str(merge_script), "--adapter_dir", str(args.adapter_dir), "--output_dir", str(args.model_dir)],
            check=True,
        )

    verify_exported_vlm_dir(args.model_dir, model_family=args.model_family, reference_dir=args.adapter_dir)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa" if args.model_family == "gemma4" else "eager",
    )
    model.eval()

    preds: list[dict] = []
    sw = sW = sc = sC = nl_e = nl_c = ot_e = ot_c = tp = pr_l = rf_l = loops = capped = non_t = leaks = mods = tleaks = 0
    raw_ce = raw_cc = 0
    n_rows = len(rows)
    cap = int(args.max_new_tokens)
    for idx, row in enumerate(rows):
        img_path = rows[(idx + 1) % n_rows]["image_path"] if (args.swap_images and n_rows > 1) else row["image_path"]
        img = Image.open(img_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cap, do_sample=False, repetition_penalty=args.repetition_penalty)
        gen_ids = out[:, inputs["input_ids"].shape[1] :]
        fr = "length" if int(gen_ids.shape[1]) >= cap else "stop"
        raw_gen = proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
        if any(t in raw_gen for t in ("<|channel>thought", "<|content_thinking|>", "<think>")):
            tleaks += 1
        gen = proc.batch_decode(gen_ids, skip_special_tokens=True)[0]
        sp = score_pair(gen, row["text"])["strict"]
        lb = score_line_breaks(gen, row["text"])
        ver = score_transcription(gen, row["text"], finish_reason=fr)
        sw += sp.word_edits; sW += sp.ref_words; sc += sp.char_edits; sC += sp.ref_chars
        nl_e += int(lb["newline_char_edits"]); nl_c += int(lb["newline_ref_chars"])
        ot_e += int(lb["order_tolerant_char_edits"]); ot_c += int(lb["order_tolerant_ref_chars"])
        tp += int(lb["matched_boundaries"]); pr_l += int(lb["pred_lines"]); rf_l += int(lb["ref_lines"])
        loops += int(ver.loop_detected); capped += int(fr == "length"); non_t += int(ver.non_terminated)
        leaks += ver.editorial_leak_count; mods += ver.modernizations
        if args.dual_convention:
            rsp = score_pair(gen, row.get("raw_text", row["text"]))["strict"]
            raw_ce += rsp.char_edits; raw_cc += rsp.ref_chars
        preds.append({
            "record_id": row["record_id"], "document_id": row["document_id"], "finish_reason": fr,
            "strict_cer": ver.cer_strict, "ref_chars": sp.ref_chars, "reward": ver.reward,
            "prediction": gen, "reference": row["text"],
        })

    if pr_l + rf_l:
        lb_f1_pct = round(200.0 * tp / (pr_l + rf_l), 2)
    else:
        all_nonempty_match = all(
            bool(normalize_preserve_newlines(p["prediction"])) == bool(normalize_preserve_newlines(p["reference"]))
            for p in preds
        )
        lb_f1_pct = 100.0 if all_nonempty_match else 0.0

    scores: dict = {
        "pages": n_rows,
        "swap_images": args.swap_images,
        "strict_wer_pct": round(100.0 * sw / sW, 2) if sW else 0.0,
        "strict_cer_pct": round(100.0 * sc / sC, 2) if sC else 0.0,
        "newline_cer_pct": round(100.0 * nl_e / nl_c, 2) if nl_c else 0.0,
        "line_boundary_f1_pct": lb_f1_pct,
        "order_tolerant_cer_pct": round(100.0 * ot_e / ot_c, 2) if ot_c else 0.0,
        "mean_reward": round(sum(p["reward"] for p in preds) / max(1, len(preds)), 4),
        "loop_pages": loops,
        "budget_capped_pages": capped,
        "non_terminated_pages": non_t,
        "editorial_leaks": leaks,
        "modernizations": mods,
        "thinking_leak_pages": tleaks,
    }
    if args.dual_convention and raw_cc:
        scores["raw_strict_cer_pct"] = round(100.0 * raw_ce / raw_cc, 2)
    if base_map is not None:
        diffs = [p["strict_cer"] - base_map[p["record_id"]] for p in preds]
        cids = [p["document_id"] for p in preds]
        wts = [max(1, int(p["ref_chars"])) for p in preds]
        scores["bootstrap_vs_baseline"] = paired_bootstrap_ci(diffs, cids, weights=wts)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"scores": scores, "predictions": preds}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
