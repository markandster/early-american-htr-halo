"""Strict and lenient diplomatic WER/CER scoring built on rapidfuzz."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

_WS_RE = re.compile(r"\s+")
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
