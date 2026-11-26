"""Shared evaluation manifest validation, scoring, thinking-leak detection, and atomic I/O."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Mapping, Sequence

from .metrics import normalize_preserve_newlines, paired_bootstrap_ci, score_line_breaks, score_pair
from .revcity_build import resolve_manifest_image_path
from .rewards import score_transcription

THINKING_LEAK_MARKERS = ("<|channel>thought", "<|content_thinking|>", "<think>")


def detect_thinking_leak(raw_text: str, reasoning_content: str | None = None) -> bool:
    """Return True if either structured ``reasoning_content`` or raw thinking markers are present."""
    if reasoning_content is not None and str(reasoning_content).strip():
        return True
    raw = str(raw_text or "")
    return any(marker in raw for marker in THINKING_LEAK_MARKERS)


def load_eval_manifest(
    eval_jsonl: Path,
    *,
    allow_frozen_test: bool = False,
    max_pages: int | None = None,
    image_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate an evaluation JSONL manifest with frozen-test and uniqueness guards."""
    rows = [json.loads(line) for line in eval_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_pages:
        rows = rows[:max_pages]
    if not rows:
        raise ValueError(f"Empty evaluation manifest: {eval_jsonl}")

    has_test_split = "test" in eval_jsonl.name.lower() or any("test" in str(r.get("split", "")).lower() for r in rows)
    if has_test_split and not allow_frozen_test:
        raise ValueError("Refusing test manifest without --allow-frozen-test during dev. Use val.jsonl.")

    eval_ids = [str(r.get("record_id") or "").strip() for r in rows]
    if any(not rid for rid in eval_ids) or len(eval_ids) != len(set(eval_ids)):
        raise ValueError(f"Missing or duplicate record_id entries found in {eval_jsonl}")

    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw_img = str(item.get("image_path") or "").strip()
        if not raw_img:
            raise ValueError(f"Missing image_path for record_id={item.get('record_id')!r} in {eval_jsonl}")
        if image_root is not None or not Path(raw_img).is_absolute():
            item["image_path"] = str(resolve_manifest_image_path(raw_img, eval_jsonl, image_root))
        resolved_rows.append(item)
    return resolved_rows


def load_baseline_cer_map(baseline_path: Path, eval_record_ids: Sequence[str]) -> dict[str, float]:
    """Load baseline predictions JSON and validate exact 1-to-1 ID parity and finite strict CER."""
    if not baseline_path.is_file():
        raise ValueError(f"Baseline predictions file not found: {baseline_path}")
    base_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_rows = base_data if isinstance(base_data, list) else base_data.get("predictions", [])
    base_ids = [str(r.get("record_id") or "").strip() for r in base_rows]
    if any(not rid for rid in base_ids) or len(base_ids) != len(set(base_ids)):
        raise ValueError(f"Missing or duplicate record_id entries found in {baseline_path}")
    if set(base_ids) != set(eval_record_ids):
        raise ValueError(
            f"Baseline predictions must match eval record_ids 1-to-1 (base={len(base_ids)}, eval={len(eval_record_ids)})"
        )
    out: dict[str, float] = {}
    try:
        for row in base_rows:
            val = float(row["strict_cer"])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite strict_cer={val!r}")
            out[str(row["record_id"]).strip()] = val
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid strict_cer entry in {baseline_path}: {exc}") from exc
    return out


def score_eval_predictions(
    rows: Sequence[Mapping[str, Any]],
    raw_predictions: Sequence[Mapping[str, Any]],
    *,
    swap_images: bool = False,
    dual_convention: bool = False,
    baseline_cer_map: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score decoded HTR predictions using corpus metrics, line breaks, and optional cluster bootstrap."""
    if len(rows) != len(raw_predictions):
        raise ValueError(f"Prediction count {len(raw_predictions)} != manifest row count {len(rows)}")

    preds: list[dict[str, Any]] = []
    sw = sW = sc = sC = nl_e = nl_c = ot_e = ot_c = tp = pr_l = rf_l = loops = capped = non_t = leaks = mods = tleaks = 0
    raw_ce = raw_cc = 0
    for row, pred in zip(rows, raw_predictions):
        gen = str(pred.get("text") or "")
        fr = str(pred.get("finish_reason") or "stop")
        raw_gen = str(pred.get("raw_text") if pred.get("raw_text") is not None else gen)
        reason_text = pred.get("reasoning_content")
        if detect_thinking_leak(raw_gen, None if reason_text is None else str(reason_text)):
            tleaks += 1

        ref_text = str(row["text"])
        sp = score_pair(gen, ref_text)["strict"]
        lb = score_line_breaks(gen, ref_text)
        ver = score_transcription(gen, ref_text, finish_reason=fr)
        sw += sp.word_edits
        sW += sp.ref_words
        sc += sp.char_edits
        sC += sp.ref_chars
        nl_e += int(lb["newline_char_edits"])
        nl_c += int(lb["newline_ref_chars"])
        ot_e += int(lb["order_tolerant_char_edits"])
        ot_c += int(lb["order_tolerant_ref_chars"])
        tp += int(lb["matched_boundaries"])
        pr_l += int(lb["pred_lines"])
        rf_l += int(lb["ref_lines"])
        loops += int(ver.loop_detected)
        capped += int(fr == "length")
        non_t += int(ver.non_terminated)
        leaks += ver.editorial_leak_count
        mods += ver.modernizations
        if dual_convention:
            rsp = score_pair(gen, str(row.get("raw_text", ref_text)))["strict"]
            raw_ce += rsp.char_edits
            raw_cc += rsp.ref_chars
        preds.append(
            {
                "record_id": row["record_id"],
                "document_id": row["document_id"],
                "finish_reason": fr,
                "strict_cer": ver.cer_strict,
                "ref_chars": sp.ref_chars,
                "reward": ver.reward,
                "prediction": gen,
                "reference": ref_text,
            }
        )

    if pr_l + rf_l:
        lb_f1_pct = round(200.0 * tp / (pr_l + rf_l), 2)
    else:
        all_nonempty_match = all(
            bool(normalize_preserve_newlines(p["prediction"])) == bool(normalize_preserve_newlines(p["reference"]))
            for p in preds
        )
        lb_f1_pct = 100.0 if all_nonempty_match else 0.0

    med_cer_pct = round(100.0 * statistics.median([float(p["strict_cer"]) for p in preds]), 2) if preds else 0.0
    scores: dict[str, Any] = {
        "pages": len(rows),
        "swap_images": swap_images,
        "strict_wer_pct": round(100.0 * sw / sW, 2) if sW else 0.0,
        "strict_cer_pct": round(100.0 * sc / sC, 2) if sC else 0.0,
        "median_strict_cer_pct": med_cer_pct,
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
    if dual_convention and raw_cc:
        scores["raw_strict_cer_pct"] = round(100.0 * raw_ce / raw_cc, 2)
    if baseline_cer_map is not None:
        diffs = [p["strict_cer"] - baseline_cer_map[str(p["record_id"])] for p in preds]
        cids = [str(p["document_id"]) for p in preds]
        wts = [max(1, int(p["ref_chars"])) for p in preds]
        scores["bootstrap_vs_baseline"] = paired_bootstrap_ci(diffs, cids, weights=wts)
    return scores, preds


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write UTF-8 JSON via temp file + ``os.replace`` inside target parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
