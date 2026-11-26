"""Strict and lenient diplomatic WER/CER scoring built on rapidfuzz."""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from rapidfuzz.distance import Levenshtein

_WS_RE = re.compile(r"\s+")
_HSPACE_RE = re.compile(r"[ \t]+")
_LENIENT_PUNCT_RE = re.compile(r"[^\w\s']+|(?<!\w)'|'(?!\w)", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Word and character edit counts and error rates for one normalization mode."""

    word_edits: int
    ref_words: int
    wer: float
    char_edits: int
    ref_chars: int
    cer: float


def normalize_strict(text: str) -> str:
    """Apply Unicode NFC normalization and collapse whitespace while preserving case and punctuation."""
    nfc = unicodedata.normalize("NFC", text or "")
    return _WS_RE.sub(" ", nfc).strip()


def normalize_lenient(text: str) -> str:
    """Apply Unicode NFC, lowercase, strip edge/standalone punctuation, and collapse whitespace."""
    nfc = unicodedata.normalize("NFC", text or "").lower()
    cleaned = _LENIENT_PUNCT_RE.sub(" ", nfc)
    return _WS_RE.sub(" ", cleaned).strip()


def _compute_breakdown(pred_norm: str, ref_norm: str) -> ScoreBreakdown:
    pred_words = pred_norm.split() if pred_norm else []
    ref_words = ref_norm.split() if ref_norm else []
    word_edits = int(Levenshtein.distance(pred_words, ref_words))
    char_edits = int(Levenshtein.distance(pred_norm, ref_norm))

    ref_w_len = len(ref_words)
    ref_c_len = len(ref_norm)

    if ref_w_len == 0:
        wer = 0.0 if len(pred_words) == 0 else 1.0
    else:
        wer = word_edits / ref_w_len

    if ref_c_len == 0:
        cer = 0.0 if len(pred_norm) == 0 else 1.0
    else:
        cer = char_edits / ref_c_len

    return ScoreBreakdown(
        word_edits=word_edits,
        ref_words=ref_w_len,
        wer=wer,
        char_edits=char_edits,
        ref_chars=ref_c_len,
        cer=cer,
    )


def score_pair(prediction: str, reference: str) -> dict[str, ScoreBreakdown]:
    """Compute both 'strict' and 'lenient' WER/CER breakdowns for a prediction-reference pair."""
    return {
        "strict": _compute_breakdown(normalize_strict(prediction), normalize_strict(reference)),
        "lenient": _compute_breakdown(normalize_lenient(prediction), normalize_lenient(reference)),
    }


def normalize_preserve_newlines(text: str) -> str:
    """Normalize NFC and horizontal spaces per line while preserving line breaks."""
    nfc = unicodedata.normalize("NFC", text or "")
    lines = [_HSPACE_RE.sub(" ", ln).strip() for ln in nfc.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _line_break_word_offsets(text: str) -> set[int]:
    """Cumulative word counts at internal line breaks (captures break placement)."""
    offsets: set[int] = set()
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    pos = 0
    for idx in range(len(lines) - 1):
        pos += len(lines[idx])
        offsets.add(pos)
    return offsets


def score_line_breaks(prediction: str, reference: str) -> dict[str, float]:
    """Compute newline-sensitive CER, line-boundary F1, and region/order-tolerant CER."""
    pred_nl = normalize_preserve_newlines(prediction)
    ref_nl = normalize_preserve_newlines(reference)
    char_edits = int(Levenshtein.distance(pred_nl, ref_nl))
    newline_cer = (char_edits / len(ref_nl)) if ref_nl else (0.0 if not pred_nl else 1.0)

    pred_breaks = _line_break_word_offsets(pred_nl)
    ref_breaks = _line_break_word_offsets(ref_nl)
    if not pred_breaks and not ref_breaks:
        boundary_f1 = 1.0 if (bool(pred_nl) == bool(ref_nl)) else 0.0
    elif not pred_breaks or not ref_breaks:
        boundary_f1 = 0.0
    else:
        tp = len(pred_breaks & ref_breaks)
        boundary_f1 = (2.0 * tp) / (len(pred_breaks) + len(ref_breaks))

    pred_lines = pred_nl.splitlines() if pred_nl else []
    ref_lines = ref_nl.splitlines() if ref_nl else []
    if not ref_lines:
        order_tolerant_cer = 0.0 if not pred_lines else 1.0
    else:
        unused = list(range(len(pred_lines)))
        matched_edits = 0
        total_ref_chars = sum(len(ln) for ln in ref_lines)
        for r_ln in ref_lines:
            if not unused:
                matched_edits += len(r_ln)
                continue
            best_i = min(unused, key=lambda idx: int(Levenshtein.distance(pred_lines[idx], r_ln)))
            matched_edits += int(Levenshtein.distance(pred_lines[best_i], r_ln))
            unused.remove(best_i)
        matched_edits += sum(len(pred_lines[idx]) for idx in unused)
        order_tolerant_cer = matched_edits / max(1, total_ref_chars)

    return {
        "newline_cer": round(newline_cer, 6),
        "line_boundary_f1": round(boundary_f1, 6),
        "order_tolerant_cer": round(order_tolerant_cer, 6),
    }


def paired_bootstrap_ci(
    diffs: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute cluster-bootstrap 95% confidence interval for paired metric differences."""
    if len(diffs) != len(cluster_ids):
        raise ValueError("diffs and cluster_ids must have identical length")
    if not diffs:
        raise ValueError("paired_bootstrap_ci requires at least one observation")

    clusters: dict[str, list[float]] = {}
    for d, cid in zip(diffs, cluster_ids, strict=True):
        clusters.setdefault(str(cid), []).append(float(d))
    keys = list(clusters)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(max(100, n_boot)):
        sampled_vals: list[float] = []
        for k in rng.choices(keys, k=len(keys)):
            sampled_vals.extend(clusters[k])
        means.append(sum(sampled_vals) / len(sampled_vals))
    means.sort()
    lo_idx = max(0, int(0.025 * len(means)))
    hi_idx = min(len(means) - 1, int(0.975 * len(means)))
    point = sum(diffs) / len(diffs)
    return {
        "mean": round(point, 6),
        "ci_low": round(means[lo_idx], 6),
        "ci_high": round(means[hi_idx], 6),
        "n_clusters": float(len(keys)),
    }
