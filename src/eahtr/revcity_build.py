"""Shared Revolutionary City corpus download, hygiene, grouping, and manifest builder."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from .candidates import (
    REVCITY_ZENODO_BYTES,
    REVCITY_ZENODO_MD5,
    assign_balanced_group_splits,
    assign_revcity_item_groups,
)
from .corpus import detect_two_page_opening, filter_corpus_record, parse_alto_xml_lines
from .hygiene import audit_split_disjointness, clean_diplomatic_target, compute_dhash, extract_word_ngrams


def file_md5(path: Path) -> str:
    """Compute hex MD5 digest of a local file in 4 MiB chunks."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_retry(
    url: str,
    dest: Path,
    *,
    max_attempts: int = 4,
    min_existing_bytes: int = 1_000_000_000,
    expected_bytes: int | None = None,
    expected_md5: str | None = None,
    progress_factory: Callable[..., Any] | None = None,
) -> Path:
    """Download ``url`` to ``dest`` with Range resume, byte-count check, and MD5 verification."""
    if dest.exists() and dest.stat().st_size >= min_existing_bytes:
        size_ok = expected_bytes is None or dest.stat().st_size == expected_bytes
        md5_ok = expected_md5 is None or file_md5(dest) == expected_md5
        if size_ok and md5_ok:
            return dest
        dest.unlink()
    if not url:
        raise ValueError(f"Archive {dest} missing or failed verification and no URL was supplied.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "early-american-htr-halo/0.1 (public-corpus-research)"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            with urlopen(Request(url, headers=headers), timeout=120) as resp:
                mode = "ab" if (existing > 0 and getattr(resp, "status", 200) == 206) else "wb"
                total = int(resp.headers.get("Content-Length") or 0) + (existing if mode == "ab" else 0)
                pbar = (
                    progress_factory(total=total or None, initial=existing if mode == "ab" else 0, unit="B", unit_scale=True, desc=dest.name)
                    if progress_factory is not None
                    else None
                )
                try:
                    with part.open(mode) as fh:
                        while chunk := resp.read(4 << 20):
                            fh.write(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))
                finally:
                    if pbar is not None and hasattr(pbar, "close"):
                        pbar.close()
            os.replace(part, dest)
            if expected_bytes is not None and dest.stat().st_size != expected_bytes:
                actual = dest.stat().st_size
                dest.unlink(missing_ok=True)
                raise OSError(f"Download byte count mismatch for {dest}: {actual} != {expected_bytes}")
            if expected_md5 is not None and file_md5(dest) != expected_md5:
                dest.unlink(missing_ok=True)
                raise OSError(f"Download MD5 mismatch for {dest}")
            return dest
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = exc
            if attempt < max_attempts:
                time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def fetch_bytes_polite(url: str, *, max_attempts: int = 3, delay_s: float = 0.5) -> bytes:
    """Fetch a small HTTP resource with polite spacing and bounded retries."""
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            time.sleep(delay_s)
            with urlopen(Request(url, headers={"User-Agent": "early-american-htr-halo/0.1 (public-corpus-research)"}), timeout=60) as resp:
                return resp.read()
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_err = exc
            if attempt < max_attempts:
                time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f"Failed GET {url}: {last_err}")


def resolve_manifest_image_path(image_path: str, manifest_path: Path, image_root: Path | None = None) -> Path:
    """Resolve relative or re-rooted manifest ``image_path`` entries; fail closed if missing."""
    raw = str(image_path or "").strip()
    if not raw:
        raise FileNotFoundError(f"Empty image_path in manifest {manifest_path}")
    candidate = Path(raw)
    if image_root is not None:
        root = image_root.resolve()
        checks = ([root / candidate] if not candidate.is_absolute() else []) + [root / candidate.name]
        for path in checks:
            if path.is_file():
                return path
        raise FileNotFoundError(f"Manifest image not found under --image-root {root}: {raw!r} (manifest={manifest_path})")
    if candidate.is_absolute():
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Absolute manifest image not found: {candidate} (manifest={manifest_path})")

    mdir = manifest_path.resolve().parent
    for base in (mdir, mdir.parent):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Relative manifest image not found: {raw!r} (manifest={manifest_path})")


def stratified_smoke_slice(
    rows_by_split: Mapping[str, Sequence[Any]],
    max_total: int,
    seed: int,
    *,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> dict[str, list[Any]]:
    """Round-robin sample pages across document groups into train/val/test_public budgets."""
    if sum(len(v) for v in rows_by_split.values()) <= max_total:
        return {k: list(v) for k, v in rows_by_split.items()}
    rng = random.Random(seed)
    out: dict[str, list[Any]] = {"train": [], "val": [], "test_public": []}
    min_eval = min(24, max(8, max_total // 10))
    budgets = {
        "val": min(len(rows_by_split["val"]), max(min_eval, int(max_total * val_fraction))),
        "test_public": min(len(rows_by_split["test_public"]), max(min_eval, int(max_total * test_fraction))),
    }
    budgets["train"] = min(len(rows_by_split["train"]), max(1, max_total - budgets["val"] - budgets["test_public"]))
    for split, k in budgets.items():
        by_group: dict[str, list[Any]] = {}
        for item in rows_by_split[split]:
            by_group.setdefault(item[6].get("document_id") or item[1], []).append(item)
        gkeys = list(by_group)
        rng.shuffle(gkeys)
        for gk in gkeys:
            rng.shuffle(by_group[gk])
        picked: list[Any] = []
        while len(picked) < k and any(by_group.values()):
            for gk in gkeys:
                if by_group[gk]:
                    picked.append(by_group[gk].pop())
                    if len(picked) >= k:
                        break
        out[split] = picked
    return out


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write JSONL records atomically via a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    os.replace(tmp, path)


def _format_record_path(img_path: Path, manifest_dir: Path, path_mode: str) -> str:
    return str(img_path.resolve()) if path_mode == "absolute" else os.path.relpath(img_path.resolve(), manifest_dir.resolve())


def _union_find_close_groups(admitted_items: Sequence[tuple[Any, ...]]) -> dict[str, str]:
    base_keys = [item[7] for item in admitted_items]
    parent = {k: k for k in base_keys}

    def _find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    seen_sig: dict[str, str] = {}
    for _, _, _, _, _, c_txt, _, b_key in admitted_items:
        sigs = set(extract_word_ngrams(c_txt, n=32))
        words_norm = " ".join(c_txt.lower().split())
        if len(words_norm.split()) >= 8:
            sigs.add("exact:" + hashlib.sha256(words_norm.encode("utf-8")).hexdigest())
        for sig in sigs:
            if sig in seen_sig:
                _union(seen_sig[sig], b_key)
            else:
                seen_sig[sig] = b_key
    return {b_key: _find(b_key) for b_key in set(base_keys)}


def tag_hard_case_overlaps(
    hard_case_rows: list[dict[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    test_public_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate ``split_overlap`` and ``panel_role`` on hard-case rows; return held-out subset."""
    train_ids = {r["document_id"] for r in train_rows}
    val_ids = {r["document_id"] for r in val_rows}
    test_ids = {r["document_id"] for r in test_public_rows}
    held_out: list[dict[str, Any]] = []
    for hc in hard_case_rows:
        did = hc.get("document_id")
        if did in test_ids:
            hc["split_overlap"], hc["panel_role"] = "test_public", "diagnostic_overlaps_test"
        elif did in train_ids:
            hc["split_overlap"], hc["panel_role"] = "train", "diagnostic_overlaps_train"
        elif did in val_ids:
            hc["split_overlap"], hc["panel_role"] = "val", "held_out_from_train_val_doc"
            held_out.append(hc)
        else:
            hc["split_overlap"], hc["panel_role"] = "none", "held_out_from_train_and_test"
            held_out.append(hc)
    return held_out


def split_summary_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize page, document, writer, and word counts for a split manifest."""
    return {
        "pages": len(rows),
        "documents": len({r["document_id"] for r in rows}),
        "attributed_hands": len({r["writer_key"] for r in rows if r.get("writer_attributed")}),
        "unattributed_pages": sum(1 for r in rows if not r.get("writer_attributed")),
        "two_page_openings": sum(1 for r in rows if r.get("is_two_page_opening")),
        "words": sum(int(r["word_count"]) for r in rows),
        "collections": dict(Counter(r["collection"] for r in rows)),
    }


def _scan_revcity_zip(
    zf: zipfile.ZipFile,
    pages_dir: Path,
    manifest_dir: Path,
    *,
    max_id_gap: int,
    creator_map: Mapping[str, str] | None,
    path_mode: str,
    corpus_report: Counter[str],
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
    by_stem: dict[str, dict[str, str]] = {}
    for member in zf.namelist():
        low, stem = member.lower(), Path(member).stem
        if low.endswith((".jpg", ".jpeg", ".png")):
            by_stem.setdefault(stem, {})["img"] = member
        elif low.endswith(".xml"):
            by_stem.setdefault(stem, {})["xml"] = member
    paired_stems = sorted(s for s, d in by_stem.items() if "img" in d and "xml" in d)
    if not paired_stems:
        raise ValueError("No paired image + ALTO-XML stems found in archive")

    item_groups = assign_revcity_item_groups(paired_stems, max_id_gap=max_id_gap, creator_map=creator_map)
    admitted: list[tuple[Any, ...]] = []
    hard_cases: list[dict[str, Any]] = []
    for stem in paired_stems:
        entry = by_stem[stem]
        xml_str = zf.read(entry["xml"]).decode("utf-8", errors="replace")
        try:
            raw_text = "\n".join(parse_alto_xml_lines(xml_str, sort_regions=False))
        except Exception:
            corpus_report["revcity_xml_parse_error"] += 1
            continue
        accepted, reason = filter_corpus_record({"source_name": "Revolutionary City", "license": "CC-BY-4.0", "text": raw_text})
        cleaned_text = clean_diplomatic_target(raw_text)
        info = item_groups.get(stem, {})
        safe_id = info.get("safe_id") or stem.replace("%3A", "_")
        if not accepted or len(cleaned_text.split()) < 8:
            drop_tag = reason if not accepted else "dropped_under_8_words"
            corpus_report[f"revcity_{drop_tag}"] += 1
            img_ext = Path(entry["img"]).suffix.lower()
            hc_img = pages_dir / f"hard_{safe_id}{img_ext}"
            if not hc_img.exists():
                hc_img.write_bytes(zf.read(entry["img"]))
            empty_alto = len(raw_text.strip()) == 0
            gt_status = "missing_or_unverified_gt" if empty_alto else ("empty_after_hygiene" if not cleaned_text.strip() else "nonblank")
            hard_cases.append({
                "record_id": f"revcity-{safe_id}",
                "document_id": info.get("document_id") or f"revcity:{safe_id}",
                "writer_key": info.get("writer_key") or f"unattributed_item:{safe_id}",
                "image_path": _format_record_path(hc_img, manifest_dir, path_mode),
                "reason": drop_tag,
                "has_empty_alto_text": empty_alto,
                "ground_truth_status": gt_status,
                "raw_text": raw_text,
                "text": cleaned_text or raw_text,
            })
            continue
        base_key = info["writer_key"] if info.get("hand_attributed") == "true" else (info.get("document_id") or f"revcity:{safe_id}")
        admitted.append((stem, safe_id, entry["img"], xml_str, raw_text, cleaned_text, info, base_key))
    if not admitted:
        raise ValueError(f"Zero admitted pages parsed from archive (paired={len(paired_stems)})")
    return admitted, hard_cases


def build_revcity_manifests(
    zip_path: Path,
    pages_dir: Path,
    manifest_dir: Path,
    *,
    max_id_gap: int = 30,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    max_eval_group_pages: int = 40,
    seed: int = 20261125,
    max_pages: int | None = None,
    creator_map: Mapping[str, str] | None = None,
    path_mode: str = "relative",
    download_url: str | None = None,
    expected_bytes: int | None = REVCITY_ZENODO_BYTES,
    expected_md5: str | None = REVCITY_ZENODO_MD5,
    min_existing_bytes: int = 1_000_000_000,
    min_eval_pages: int = 8,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Scan RevCity ZIP, enforce group/text closure and disjointness, extract images, and write manifests."""
    if path_mode not in ("relative", "absolute"):
        raise ValueError(f"path_mode must be 'relative' or 'absolute', got {path_mode!r}")
    pages_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if download_url:
        download_with_retry(download_url, zip_path, min_existing_bytes=min_existing_bytes, expected_bytes=expected_bytes, expected_md5=expected_md5)
    else:
        if not zip_path.is_file():
            raise FileNotFoundError(f"RevCity archive not found: {zip_path}")
        if expected_bytes is not None and zip_path.stat().st_size != expected_bytes:
            raise ValueError(f"RevCity archive byte mismatch: {zip_path.stat().st_size} != {expected_bytes}")
        if expected_md5 is not None and file_md5(zip_path) != expected_md5:
            raise ValueError(f"RevCity archive MD5 mismatch for {zip_path}")

    corpus_report: Counter[str] = Counter()
    all_records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test_public": []}
    with zipfile.ZipFile(zip_path) as zf:
        admitted, hard_case_rows = _scan_revcity_zip(
            zf, pages_dir, manifest_dir, max_id_gap=max_id_gap, creator_map=creator_map, path_mode=path_mode, corpus_report=corpus_report
        )
        root_by_base = _union_find_close_groups(admitted)
        group_stats: dict[str, dict[str, Any]] = {}
        for _, _, _, _, _, _, info, b_key in admitted:
            ent = group_stats.setdefault(root_by_base[b_key], {"pages": 0, "institution": info.get("institution") or "revcity"})
            ent["pages"] += 1
        split_map = assign_balanced_group_splits(group_stats, val_fraction=val_fraction, test_fraction=test_fraction, max_eval_group_pages=max_eval_group_pages, seed=seed)
        split_buckets: dict[str, list[Any]] = {"train": [], "val": [], "test_public": []}
        for stem, safe_id, img_m, xml_str, raw_text, cleaned_text, info, base_key in admitted:
            split_buckets[split_map[root_by_base[base_key]]].append((stem, safe_id, img_m, xml_str, raw_text, cleaned_text, info))
        if max_pages is not None:
            split_buckets = stratified_smoke_slice(split_buckets, max_pages, seed, val_fraction=val_fraction, test_fraction=test_fraction)
        for split, items in split_buckets.items():
            for stem, safe_id, img_member, xml_str, raw_text, cleaned_text, info in items:
                img_ext = Path(img_member).suffix.lower()
                img_bytes = zf.read(img_member)
                img_out = pages_dir / f"{safe_id}{img_ext}"
                if not img_out.exists():
                    img_out.write_bytes(img_bytes)
                all_records_by_split[split].append({
                    "record_id": f"revcity-{safe_id}",
                    "collection": "revcity-zenodo-15776323",
                    "document_id": info.get("document_id") or f"revcity:{safe_id}",
                    "writer_key": info.get("writer_key") or f"unattributed_item:{safe_id}",
                    "writer_attributed": info.get("hand_attributed") == "true",
                    "is_two_page_opening": detect_two_page_opening(xml_str),
                    "image_format": img_ext.lstrip("."),
                    "image_path": _format_record_path(img_out, manifest_dir, path_mode),
                    "image_sha256": hashlib.sha256(img_bytes).hexdigest(),
                    "image_dhash": compute_dhash(img_out),
                    "raw_text": raw_text,
                    "text": cleaned_text,
                    "word_count": len(cleaned_text.split()),
                    "license": "CC-BY-4.0",
                    "split": split,
                })
                corpus_report[f"revcity_admitted_{split}"] += 1
                corpus_report[f"revcity_ext_{img_ext.lstrip('.')}"] += 1

    train_rows, val_rows, test_rows = all_records_by_split["train"], all_records_by_split["val"], all_records_by_split["test_public"]
    if len(train_rows) < 1 or len(val_rows) < min_eval_pages or len(test_rows) < min_eval_pages:
        raise ValueError(f"Splits too small: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    val_audit = audit_split_disjointness(train_rows, val_rows, ngram_size=32, max_dhash_distance=4)
    test_audit = audit_split_disjointness(train_rows, test_rows, ngram_size=32, max_dhash_distance=4)
    val_test_audit = audit_split_disjointness(val_rows, test_rows, ngram_size=32, max_dhash_distance=4)
    for label, audit in (("Train/Val", val_audit), ("Train/Test", test_audit), ("Val/Test", val_test_audit)):
        if not audit["passed"]:
            raise AssertionError(f"{label} leakage detected: {audit}")

    held_out = tag_hard_case_overlaps(hard_case_rows, train_rows, val_rows, test_rows)
    all_manifests: dict[str, list[dict[str, Any]]] = {**all_records_by_split, "hard_case_panel": hard_case_rows, "hard_case_held_out": held_out}
    if write_outputs:
        for split_name, rows in all_manifests.items():
            write_jsonl_atomic(manifest_dir / f"{split_name}.jsonl", rows)
    summary = {
        "counts": dict(corpus_report),
        "splits": {k: split_summary_stats(v) for k, v in all_records_by_split.items()},
        "hard_case_panel_pages": len(hard_case_rows),
        "hard_case_held_out_pages": len(held_out),
        "dhash_review_candidates": val_audit["perceptual_image_collisions"] + test_audit["perceptual_image_collisions"] + val_test_audit["perceptual_image_collisions"],
        "val_disjoint_audit": val_audit,
        "test_disjoint_audit": test_audit,
        "val_test_disjoint_audit": val_test_audit,
        "path_mode": path_mode,
        "manifests": all_manifests,
    }
    if write_outputs:
        (manifest_dir.parent / "corpus_audit_report.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "manifests"}, indent=2), encoding="utf-8")
    return summary
