#!/usr/bin/env python3
"""Generate K=4 Stage 1 rollouts and build gold-anchored Stage 2 SMPO preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from eahtr.candidates import DIPLOMATIC_PROMPT
from eahtr.preferences import build_preference_pairs
from eahtr.rewards import recommended_max_new_tokens, score_group_completions


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample Stage 1 rollouts and build Halo SMPO preference JSONL.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Merged Stage 1 SFT checkpoint directory")
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Train manifest JSONL")
    parser.add_argument("--out-pref-jsonl", type=Path, required=True, help="Output preference JSONL for export_halo_datasets.py")
    parser.add_argument("--out-rollouts-jsonl", type=Path, default=None, help="Optional raw scored rollout log")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--full-train-count", type=int, default=3000, help="Count for throughput/GPU-hour estimate")
    parser.add_argument("--min-reward-margin", type=float, default=0.05)
    parser.add_argument("--source-fold", default="oof", help="Fold tag recorded for preference provenance")
    parser.add_argument("--allow-eval-split", action="store_true", help="Allow val/test manifests explicitly (not for Stage-2 train export)")
    args = parser.parse_args()

    import sys
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = [json.loads(line) for line in args.train_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    has_non_train = any(str(r.get("split") or "").strip().lower() != "train" for r in rows)
    has_forbidden_name = any(tok in args.train_jsonl.name.lower() for tok in ("val", "test", "hard_case"))
    if (has_non_train or has_forbidden_name) and not args.allow_eval_split:
        print(
            f"Refusing non-train or eval/hard_case manifest {args.train_jsonl} for Stage-2 rollout generation; pass out-of-fold split='train' shards or --allow-eval-split.",
            file=sys.stderr,
        )
        return 2
    if args.max_prompts:
        rows = rows[: args.max_prompts]

    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    tok = getattr(proc, "tokenizer", proc)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model.eval()

    temps = (0.3, 0.7, 0.7, 1.0)
    pref_records: list[dict] = []
    rollout_rows: list[dict] = []
    t0 = time.time()
    for row in rows:
        img = Image.open(row["image_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIPLOMATIC_PROMPT}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=[img], return_tensors="pt").to(model.device)
        cap = recommended_max_new_tokens(len(tok(row["text"], add_special_tokens=False)["input_ids"]), min_tokens=2048)
        comps, frs = [], []
        for idx in range(args.k):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=cap, do_sample=True, temperature=temps[idx % len(temps)], top_p=0.9)
            gen_ids = out[:, inputs["input_ids"].shape[1] :]
            frs.append("length" if int(gen_ids.shape[1]) >= cap else "stop")
            comps.append(proc.batch_decode(gen_ids, skip_special_tokens=True)[0])
        scored = score_group_completions(comps, row["text"], frs)
        rollout_rows.append({"record_id": row["record_id"], "completions": comps, "rewards": [s["reward"] for s in scored]})
        pref_records.append({
            "record_id": row["record_id"], "document_id": row["document_id"], "writer_key": row["writer_key"],
            "image_sha256": row.get("image_sha256", ""), "image_dhash": row.get("image_dhash", ""),
            "image_path": row["image_path"], "prompt": DIPLOMATIC_PROMPT, "text": row["text"],
            "completions": comps, "finish_reasons": frs, "max_new_tokens_cap": cap,
            "split": row.get("split", "train"), "source_fold": args.source_fold, "source_checkpoint": str(args.model_dir),
        })

    elapsed = max(0.01, time.time() - t0)
    sec_per_page = elapsed / max(1, len(rows))
    est_gpu_hours = round(sec_per_page * args.full_train_count / 3600.0, 2)
    pairs = build_preference_pairs(pref_records, min_reward_margin=args.min_reward_margin, prefer_gold_chosen=True)

    args.out_pref_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_pref_jsonl.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + ("\n" if pairs else ""), encoding="utf-8"
    )
    if args.out_rollouts_jsonl:
        args.out_rollouts_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.out_rollouts_jsonl.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rollout_rows) + ("\n" if rollout_rows else ""), encoding="utf-8"
        )
    print(json.dumps({
        "prompts_sampled": len(rows), "pairs_emitted": len(pairs),
        "sec_per_page": round(sec_per_page, 2), "est_gpu_hours_full": est_gpu_hours,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
