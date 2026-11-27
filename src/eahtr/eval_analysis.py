"""Diagnostic error analysis for diplomatic HTR predictions and line-break drift."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import (
    normalize_preserve_newlines,
    paired_bootstrap_ci,
    score_line_breaks,
    score_pair,
)
from .rewards import score_transcription


def load_predictions_json(path: Path) -> list[dict[str, Any]]:
    """Load per-page prediction dicts from a raw list or a report wrapper JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, Mapping) and isinstance(data.get("predictions"), list):
        return [dict(item) for item in data["predictions"]]
    raise ValueError(f"Unsupported predictions JSON structure in {path}")


def _line_break_offsets(text: str) -> set[int]:
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    offsets: set[int] = set()
    pos = 0
    for idx in range(len(lines) - 1):
        pos += len(lines[idx])
        offsets.add(pos)
    return offsets


def aligned_line_boundary_f1(prediction: str, reference: str) -> float:
    """Compute line-boundary F1 after aligning predicted words to reference words."""
    pred_nl = normalize_preserve_newlines(prediction)
    ref_nl = normalize_preserve_newlines(reference)
    pred_breaks = _line_break_offsets(pred_nl)
    ref_breaks = _line_break_offsets(ref_nl)
    if not pred_breaks and not ref_breaks:
        return 1.0 if (bool(pred_nl) == bool(ref_nl)) else 0.0
    if not pred_breaks or not ref_breaks:
        return 0.0

    pred_words = pred_nl.split()
    ref_words = ref_nl.split()
    offset_map: dict[int, int] = {0: 0, len(pred_words): len(ref_words)}
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, pred_words, ref_words, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1 + 1):
                offset_map[i1 + k] = j1 + k
        else:
            p_span = max(1, i2 - i1)
            for k in range(p_span + 1):
                offset_map.setdefault(i1 + k, j1 + round(k * (j2 - j1) / p_span))

    mapped_pred = {offset_map.get(pos, round(pos * len(ref_words) / max(1, len(pred_words)))) for pos in pred_breaks}
    matched = len(mapped_pred & ref_breaks)
    return round((2.0 * matched) / (len(mapped_pred) + len(ref_breaks)), 6)


def _baseline_cer_lookup(baseline_predictions: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not baseline_predictions:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in baseline_predictions:
        rid = str(row.get("record_id") or "").strip()
        if rid:
            out[rid] = dict(row)
    return out


def analyze_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    baseline_predictions: Sequence[Mapping[str, Any]] | None = None,
    manifest_rows: Sequence[Mapping[str, Any]] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Summarize aggregate metrics, line-break offset drift, loop/cap failures, and opening slices."""
    if not predictions:
        raise ValueError("analyze_predictions requires at least one prediction row")

    manifest_by_id = {str(r.get("record_id") or "").strip(): dict(r) for r in (manifest_rows or ()) if r.get("record_id")}
    baseline_by_id = _baseline_cer_lookup(baseline_predictions)

    sw = sW = sc = sC = lw = lW = lc = lC = 0
    nl_e = nl_c = ot_e = ot_c = lb_tp = lb_p = lb_r = 0
    aligned_f1s: list[float] = []
    rewards: list[float] = []
    loops = capped = non_terms = leaks = mods = 0
    exact_line_count_pages = shifted_offset_pages = 0
    abs_line_diffs: list[int] = []
    per_page: list[dict[str, Any]] = []
    loop_or_capped: list[dict[str, Any]] = []
    opening_buckets: dict[str, list[int]] = {"two_page_opening": [0, 0, 0], "single_page": [0, 0, 0]}

    pair_diffs: list[float] = []
    pair_clusters: list[str] = []
    pair_weights: list[float] = []
    improved_pages = regressed_pages = unchanged_pages = 0

    for row in predictions:
        rid = str(row.get("record_id") or "").strip()
        m_row = manifest_by_id.get(rid, {})
        pred_text = str(row.get("prediction") if row.get("prediction") is not None else row.get("text") or "")
        ref_text = str(row.get("reference") if row.get("reference") is not None else m_row.get("text") or "")
        fr = str(row.get("finish_reason") or "stop")
        doc_id = str(row.get("document_id") or m_row.get("document_id") or rid)

        pair = score_pair(pred_text, ref_text)
        st, ln = pair["strict"], pair["lenient"]
        sw += st.word_edits
        sW += st.ref_words
        sc += st.char_edits
        sC += st.ref_chars
        lw += ln.word_edits
        lW += ln.ref_words
        lc += ln.char_edits
        lC += ln.ref_chars

        lb = score_line_breaks(pred_text, ref_text)
        nl_e += int(lb["newline_char_edits"])
        nl_c += int(lb["newline_ref_chars"])
        ot_e += int(lb["order_tolerant_char_edits"])
        ot_c += int(lb["order_tolerant_ref_chars"])
        lb_tp += int(lb["matched_boundaries"])
        lb_p += int(lb["pred_lines"])
        lb_r += int(lb["ref_lines"])

        ver = score_transcription(pred_text, ref_text, finish_reason=fr)
        rewards.append(ver.reward)
        loops += int(ver.loop_detected)
        capped += int(fr == "length")
        non_terms += int(ver.non_terminated)
        leaks += ver.editorial_leak_count
        mods += ver.modernizations

        ref_nl = normalize_preserve_newlines(ref_text)
        pred_nl = normalize_preserve_newlines(pred_text)
        ref_n_lines = len(ref_nl.splitlines()) if ref_nl else 0
        pred_n_lines = len(pred_nl.splitlines()) if pred_nl else 0
        line_diff = pred_n_lines - ref_n_lines
        abs_line_diffs.append(abs(line_diff))
        if line_diff == 0:
            exact_line_count_pages += 1
        aligned_f1 = aligned_line_boundary_f1(pred_text, ref_text)
        aligned_f1s.append(aligned_f1)
        if abs(line_diff) <= 2 and (lb["newline_cer"] - st.cer) <= 0.03 and lb["line_boundary_f1"] < 0.30:
            shifted_offset_pages += 1

        if m_row:
            key = "two_page_opening" if m_row.get("is_two_page_opening") else "single_page"
            opening_buckets[key][0] += st.char_edits
            opening_buckets[key][1] += st.ref_chars
            opening_buckets[key][2] += 1

        if rid in baseline_by_id:
            b_row = baseline_by_id[rid]
            b_pred = str(b_row.get("prediction") if b_row.get("prediction") is not None else b_row.get("text") or "")
            b_cer = float(b_row["strict_cer"]) if "strict_cer" in b_row else score_pair(b_pred, ref_text)["strict"].cer
            diff = st.cer - b_cer
            pair_diffs.append(diff)
            pair_clusters.append(doc_id)
            pair_weights.append(float(max(1, st.ref_chars)))
            if diff < -1e-6:
                improved_pages += 1
            elif diff > 1e-6:
                regressed_pages += 1
            else:
                unchanged_pages += 1

        page_info = {
            "record_id": rid,
            "document_id": doc_id,
            "strict_cer_pct": round(100.0 * st.cer, 2),
            "lenient_cer_pct": round(100.0 * ln.cer, 2),
            "newline_cer_pct": round(100.0 * lb["newline_cer"], 2),
            "order_tolerant_cer_pct": round(100.0 * lb["order_tolerant_cer"], 2),
            "line_boundary_f1_pct": round(100.0 * lb["line_boundary_f1"], 2),
            "aligned_line_boundary_f1_pct": round(100.0 * aligned_f1, 2),
            "ref_chars": st.ref_chars,
            "pred_chars": len(pred_text),
            "ref_lines": ref_n_lines,
            "pred_lines": pred_n_lines,
            "finish_reason": fr,
            "loop_detected": ver.loop_detected,
            "reward": ver.reward,
            "ref_preview": ref_text[:180],
            "pred_preview": pred_text[:180],
        }
        per_page.append(page_info)
        if ver.loop_detected or fr == "length":
            loop_or_capped.append({**page_info, "pred_tail": pred_text[-240:]})

    lb_f1 = ((2.0 * lb_tp) / (lb_p + lb_r)) if (lb_p + lb_r) else (
        1.0 if all(bool(normalize_preserve_newlines(str(r.get("prediction") or ""))) == bool(normalize_preserve_newlines(str(r.get("reference") or ""))) for r in predictions) else 0.0
    )
    summary: dict[str, Any] = {
        "pages": len(predictions),
        "scores": {
            "strict_wer_pct": round(100.0 * sw / sW, 2) if sW else 0.0,
            "strict_cer_pct": round(100.0 * sc / sC, 2) if sC else 0.0,
            "lenient_wer_pct": round(100.0 * lw / lW, 2) if lW else 0.0,
            "lenient_cer_pct": round(100.0 * lc / lC, 2) if lC else 0.0,
            "newline_cer_pct": round(100.0 * nl_e / nl_c, 2) if nl_c else 0.0,
            "line_boundary_f1_pct": round(100.0 * lb_f1, 2),
            "aligned_line_boundary_f1_pct": round(100.0 * sum(aligned_f1s) / max(1, len(aligned_f1s)), 2),
            "order_tolerant_cer_pct": round(100.0 * ot_e / ot_c, 2) if ot_c else 0.0,
            "mean_verifiable_reward": round(sum(rewards) / max(1, len(rewards)), 4),
            "loop_pages": loops,
            "budget_capped_pages": capped,
            "non_terminated_pages": non_terms,
            "editorial_leaks": leaks,
            "modernizations": mods,
        },
        "line_break_diagnostics": {
            "mean_abs_line_count_diff": round(sum(abs_line_diffs) / max(1, len(abs_line_diffs)), 2),
            "exact_line_count_pages": exact_line_count_pages,
            "shifted_offset_pages": shifted_offset_pages,
            "raw_boundary_f1_pct": round(100.0 * lb_f1, 2),
            "word_aligned_boundary_f1_pct": round(100.0 * sum(aligned_f1s) / max(1, len(aligned_f1s)), 2),
        },
        "loop_or_capped_pages": loop_or_capped,
        "worst_pages": sorted(per_page, key=lambda p: (-p["strict_cer_pct"], p["record_id"]))[: max(1, top_k)],
    }
    if manifest_by_id:
        summary["opening_breakdown"] = {
            name: {
                "pages": cnt,
                "strict_cer_pct": round(100.0 * edits / chars, 2) if chars else 0.0,
            }
            for name, (edits, chars, cnt) in opening_buckets.items()
        }
    if pair_diffs:
        ci = paired_bootstrap_ci(pair_diffs, pair_clusters, weights=pair_weights)
        summary["baseline_comparison"] = {
            "matched_pages": len(pair_diffs),
            "improved_pages": improved_pages,
            "regressed_pages": regressed_pages,
            "unchanged_pages": unchanged_pages,
            "mean_cer_diff_pp": round(100.0 * ci["mean"], 2),
            "ci95_low_pp": round(100.0 * ci["ci_low"], 2),
            "ci95_high_pp": round(100.0 * ci["ci_high"], 2),
        }
    return summary
