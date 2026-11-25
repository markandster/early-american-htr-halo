"""Target text hygiene, copy-slip filtering, and writer-disjoint split auditing."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_BRACKET_SPAN_RE = re.compile(r"\[[^\]]*\]")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_LEADER_DOTS_RE = re.compile(r"\.{6,}|(?:\.\s){5,}\.")


def clean_diplomatic_target(
    text: str,
    *,
    strip_brackets: bool = True,
    normalize_long_s: bool = True,
    strip_caret_markers: bool = True,
) -> str:
    """Normalize a diplomatic ground-truth page while preserving line breaks and period spelling."""
    normalized = unicodedata.normalize("NFC", text or "")
    if normalize_long_s:
        normalized = normalized.replace("ſ", "s")
    if strip_caret_markers:
        normalized = normalized.replace("^", "")

    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line
        if strip_brackets:
            line = _BRACKET_SPAN_RE.sub(" ", line)
        line = _MULTISPACE_RE.sub(" ", line).strip()
        # Drop malformed standalone hyphen-only lines that can induce absorbing '-' loops
        if not line or line == "-":
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def has_adjacent_duplicate_lines(text: str, min_chars: int = 12) -> bool:
    """Return True if two adjacent lines are identical and at least min_chars long."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for idx in range(1, len(lines)):
        if len(lines[idx]) >= min_chars and lines[idx] == lines[idx - 1]:
            return True
    return False


def has_table_leader_dots(text: str) -> bool:
    """Detect tabular leader-dot sequences that commonly cause token-limit repetition loops."""
    return bool(_LEADER_DOTS_RE.search(text or ""))


def extract_word_ngrams(text: str, n: int = 32) -> set[str]:
    """Return SHA-256 hashes of normalized whitespace-delimited n-grams."""
    if n <= 0:
        raise ValueError("n must be positive")
    words = unicodedata.normalize("NFC", text or "").lower().split()
    if len(words) < n:
        return set()
    hashes: set[str] = set()
    for i in range(len(words) - n + 1):
        span = " ".join(words[i : i + n])
        hashes.add(hashlib.sha256(span.encode("utf-8")).hexdigest())
    return hashes


def audit_split_disjointness(
    train_records: list[dict],
    eval_records: list[dict],
    ngram_size: int = 32,
) -> dict[str, object]:
    """Verify zero overlap across document_id, writer_key, image_sha256, and text n-grams."""
    train_docs = {str(r["document_id"]) for r in train_records if r.get("document_id")}
    train_writers = {str(r["writer_key"]) for r in train_records if r.get("writer_key")}
    train_images = {str(r["image_sha256"]) for r in train_records if r.get("image_sha256")}
    train_ngrams: set[str] = set()
    for rec in train_records:
        train_ngrams.update(extract_word_ngrams(str(rec.get("text", "")), n=ngram_size))

    eval_docs = {str(r["document_id"]) for r in eval_records if r.get("document_id")}
    eval_writers = {str(r["writer_key"]) for r in eval_records if r.get("writer_key")}
    eval_images = {str(r["image_sha256"]) for r in eval_records if r.get("image_sha256")}
    eval_ngrams: set[str] = set()
    for rec in eval_records:
        eval_ngrams.update(extract_word_ngrams(str(rec.get("text", "")), n=ngram_size))

    doc_collisions = sorted(train_docs & eval_docs)
    writer_collisions = sorted(train_writers & eval_writers)
    image_collisions = sorted(train_images & eval_images)
    ngram_collisions = len(train_ngrams & eval_ngrams)

    passed = not (doc_collisions or writer_collisions or image_collisions or ngram_collisions)
    return {
        "passed": passed,
        "doc_collisions": doc_collisions,
        "writer_collisions": writer_collisions,
        "image_collisions": image_collisions,
        "ngram_collisions": ngram_collisions,
    }
