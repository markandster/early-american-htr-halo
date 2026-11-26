"""Target text hygiene, copy-slip filtering, and writer-disjoint split auditing."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_INTRAWORD_DAMAGE_RE = re.compile(r"(?<=\w)\[_+\](?=\w)")
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
            # Collapse intra-word illegible damage (e.g. m[_]ung -> mung) before spacing out deletions.
            line = _INTRAWORD_DAMAGE_RE.sub("", line)
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


def _full_page_text_hash(text: str, min_words: int = 8) -> str | None:
    """Return normalized full-page SHA-256 hash when page has at least min_words words."""
    words = unicodedata.normalize("NFC", text or "").lower().split()
    if len(words) < min_words:
        return None
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()


def compute_dhash(image_or_path: object, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash (hex) to catch rescans and alternate crops."""
    from PIL import Image

    if hasattr(image_or_path, "convert"):
        gray = image_or_path.convert("L").resize((hash_size + 1, hash_size))
        bits = 0
        pixels = list(gray.getdata())
    else:
        with Image.open(image_or_path) as img:  # type: ignore[arg-type]
            gray = img.convert("L").resize((hash_size + 1, hash_size))
            pixels = list(gray.getdata())
        bits = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[row_start + col] < pixels[row_start + col + 1])
    return f"{bits:0{hash_size * hash_size // 4}x}"


def dhash_hamming_distance(hex_a: str, hex_b: str) -> int:
    """Return bit Hamming distance between two hex dHash strings."""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def audit_split_disjointness(
    train_records: list[dict],
    eval_records: list[dict],
    ngram_size: int = 32,
    max_dhash_distance: int = 4,
) -> dict[str, object]:
    """Verify zero overlap across document_id, writer_key, image hashes, dHashes, and n-grams."""
    train_docs = {str(r["document_id"]) for r in train_records if r.get("document_id")}
    train_writers = {str(r["writer_key"]) for r in train_records if r.get("writer_key")}
    train_images = {str(r["image_sha256"]) for r in train_records if r.get("image_sha256")}
    train_dhashes = [
        (str(r.get("record_id", idx)), str(r["image_dhash"]))
        for idx, r in enumerate(train_records)
        if r.get("image_dhash")
    ]
    train_ngrams: set[str] = set()
    train_exact_hashes: set[str] = set()
    for rec in train_records:
        text = str(rec.get("text", ""))
        train_ngrams.update(extract_word_ngrams(text, n=ngram_size))
        page_hash = _full_page_text_hash(text)
        if page_hash is not None:
            train_exact_hashes.add(page_hash)

    eval_docs = {str(r["document_id"]) for r in eval_records if r.get("document_id")}
    eval_writers = {str(r["writer_key"]) for r in eval_records if r.get("writer_key")}
    eval_images = {str(r["image_sha256"]) for r in eval_records if r.get("image_sha256")}
    eval_dhashes = [
        (str(r.get("record_id", idx)), str(r["image_dhash"]))
        for idx, r in enumerate(eval_records)
        if r.get("image_dhash")
    ]
    eval_ngrams: set[str] = set()
    eval_exact_hashes: set[str] = set()
    for rec in eval_records:
        text = str(rec.get("text", ""))
        eval_ngrams.update(extract_word_ngrams(text, n=ngram_size))
        page_hash = _full_page_text_hash(text)
        if page_hash is not None:
            eval_exact_hashes.add(page_hash)

    doc_collisions = sorted(train_docs & eval_docs)
    writer_collisions = sorted(train_writers & eval_writers)
    image_collisions = sorted(train_images & eval_images)
    ngram_collisions = len(train_ngrams & eval_ngrams)
    exact_text_collisions = len(train_exact_hashes & eval_exact_hashes)
    perceptual_collisions: list[tuple[str, str, int]] = []
    for eid, eh in eval_dhashes:
        for tid, th in train_dhashes:
            dist = dhash_hamming_distance(th, eh)
            if dist <= max_dhash_distance:
                perceptual_collisions.append((tid, eid, dist))

    passed = not (
        doc_collisions
        or writer_collisions
        or image_collisions
        or ngram_collisions
        or exact_text_collisions
        or perceptual_collisions
    )
    return {
        "passed": passed,
        "doc_collisions": doc_collisions,
        "writer_collisions": writer_collisions,
        "image_collisions": image_collisions,
        "perceptual_image_collisions": perceptual_collisions,
        "ngram_collisions": ngram_collisions,
        "exact_text_collisions": exact_text_collisions,
    }
