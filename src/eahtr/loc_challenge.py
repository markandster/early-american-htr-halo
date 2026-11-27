"""Standalone Library of Congress By the People (LCCN 2020446971) challenge builder."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

from .candidates import LOC_WASHINGTON_ITEM_JSON_URL
from .corpus import filter_corpus_record
from .hygiene import audit_split_disjointness, clean_diplomatic_target, compute_dhash
from .revcity_build import download_with_retry, fetch_bytes_polite, split_summary_stats, write_jsonl_atomic


def _find_loc_zip_url(item_meta: Mapping[str, Any]) -> str:
    zip_urls = [
        str(sub["url"])
        for res in item_meta.get("resources", [])
        for f_entry in res.get("files", [])
        for sub in (f_entry if isinstance(f_entry, list) else [f_entry])
        if isinstance(sub, Mapping) and str(sub.get("url", "")).lower().endswith(".zip")
    ]
    if not zip_urls:
        raise ValueError("Could not find Public Domain dataset .zip URL in LOC item metadata")
    return zip_urls[0]


def _format_loc_image_path(img_out: Path, manifest_dir: Path, path_mode: str) -> str:
    if path_mode == "absolute":
        return str(img_out.resolve())
    return os.path.relpath(img_out.resolve(), manifest_dir.resolve())


def build_loc_challenge_manifest(
    out_dir: Path,
    *,
    max_pages: int | None = 150,
    path_mode: str = "relative",
    zip_path: Path | None = None,
    item_json_url: str = LOC_WASHINGTON_ITEM_JSON_URL,
    image_delay_s: float = 0.4,
    revcity_splits: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Download, filter, cache, and audit LOC By the People (Public Domain) eval-only pages."""
    if path_mode not in ("relative", "absolute"):
        raise ValueError(f"path_mode must be 'relative' or 'absolute', got {path_mode!r}")

    pages_dir = out_dir / "pages"
    manifest_dir = out_dir / "manifests"
    pages_dir.mkdir(parents=True, exist_ok=True)
    if write_outputs:
        manifest_dir.mkdir(parents=True, exist_ok=True)

    target_zip = zip_path if zip_path is not None else (out_dir / "2020446971.zip")
    if not target_zip.is_file():
        item_meta = json.loads(fetch_bytes_polite(item_json_url, delay_s=0.2).decode("utf-8"))
        download_with_retry(_find_loc_zip_url(item_meta), target_zip, min_existing_bytes=1_000)

    counts: Counter[str] = Counter()
    challenge_rows: list[dict[str, Any]] = []
    hard_case_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(target_zip) as zf:
        csv_names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not csv_names:
            raise ValueError(f"No CSV found inside LOC By the People zip: {target_zip}")
        reader = csv.DictReader(io.StringIO(zf.read(csv_names[0]).decode("utf-8", errors="replace")))
        for row_idx, row in enumerate(reader):
            if max_pages is not None and len(challenge_rows) >= max_pages:
                break
            item_title = (row.get("Item") or row.get("Title") or "").lower()
            if "farm report" in item_title:
                counts["loc_skipped_farm_report"] += 1
                continue
            raw_text = row.get("Transcription") or row.get("Text") or ""
            img_url = (row.get("DownloadUrl") or row.get("Asset") or "").strip()
            if not raw_text.strip() or not img_url.startswith("http"):
                counts["loc_skipped_missing_fields"] += 1
                continue

            accepted, reason = filter_corpus_record(
                {"source_name": "LOC By the People", "license": "Public Domain", "text": raw_text}
            )
            cleaned_text = clean_diplomatic_target(raw_text)
            rec_id = f"loc-btp-{row_idx:05d}"
            doc_group = str(row.get("ItemId") or row.get("Item") or rec_id)
            if not accepted or len(cleaned_text.split()) < 8:
                drop_tag = reason if not accepted else "under_8_words"
                counts[f"loc_dropped_{drop_tag}"] += 1
                has_empty = len(raw_text.strip()) == 0
                hard_case_rows.append(
                    {
                        "record_id": rec_id,
                        "document_id": f"loc:{doc_group}",
                        "writer_key": f"loc_item:{doc_group}",
                        "reason": drop_tag,
                        "has_empty_alto_text": has_empty,
                        "ground_truth_status": "missing_or_unverified_gt" if has_empty else drop_tag,
                        "raw_text": raw_text,
                        "text": cleaned_text or raw_text,
                    }
                )
                continue

            img_out = pages_dir / f"{rec_id}.jpg"
            if img_out.is_file() and img_out.stat().st_size > 0:
                jpg_bytes = img_out.read_bytes()
            else:
                try:
                    jpg_bytes = fetch_bytes_polite(img_url, delay_s=image_delay_s)
                except Exception:
                    counts["loc_image_fetch_error"] += 1
                    continue
                img_out.write_bytes(jpg_bytes)

            challenge_rows.append(
                {
                    "record_id": rec_id,
                    "collection": "loc-by-the-people-2020446971",
                    "document_id": f"loc:{doc_group}",
                    "writer_key": f"loc_item:{doc_group}",
                    "writer_attributed": False,
                    "image_path": _format_loc_image_path(img_out, manifest_dir, path_mode),
                    "image_sha256": hashlib.sha256(jpg_bytes).hexdigest(),
                    "image_dhash": compute_dhash(img_out),
                    "raw_text": raw_text,
                    "text": cleaned_text,
                    "word_count": len(cleaned_text.split()),
                    "license": "Public Domain",
                    "split": "challenge_loc",
                }
            )
            counts["loc_challenge_admitted"] += 1

    cross_audits: dict[str, Any] = {}
    if revcity_splits:
        for split_name, split_rows in revcity_splits.items():
            audit = audit_split_disjointness(split_rows, challenge_rows, ngram_size=32, max_dhash_distance=4)
            cross_audits[split_name] = audit
            if not audit["passed"]:
                raise AssertionError(f"Cross-corpus leakage between RevCity {split_name} and LOC challenge: {audit}")

    doc_counts = dict(Counter(r["document_id"] for r in challenge_rows))
    audit_payload: dict[str, Any] = {
        "collection": "loc-by-the-people-2020446971",
        "license": "Public Domain",
        "path_mode": path_mode,
        "counts": dict(counts),
        "challenge_loc": split_summary_stats(challenge_rows) if challenge_rows else {"pages": 0, "documents": 0, "words": 0},
        "document_breakdown": doc_counts,
        "hard_case_pages": len(hard_case_rows),
        "cross_corpus_audits": cross_audits,
    }
    if write_outputs:
        write_jsonl_atomic(manifest_dir / "challenge_loc.jsonl", challenge_rows)
        write_jsonl_atomic(manifest_dir / "challenge_loc_hard_cases.jsonl", hard_case_rows)
        (out_dir / "challenge_loc_audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    return {
        "challenge_rows": challenge_rows,
        "hard_case_rows": hard_case_rows,
        "counts": dict(counts),
        "audit": audit_payload,
    }
