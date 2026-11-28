"""Merge and leakage-audit multi-fold Stage 1 OOF preference JSONL shards."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from .revcity_build import write_jsonl_atomic

_FORBIDDEN_FOLDS = {"val", "validation", "test", "test_public", "hard_case"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_and_audit_oof_preferences(
    input_paths: Sequence[Path],
    out_jsonl: Path,
    *,
    eval_manifests: Sequence[Path] = (),
    min_reward_margin: float = 0.05,
    require_distinct_folds: bool = False,
) -> dict[str, Any]:
    """Merge multi-fold OOF preference JSONLs, enforce provenance/margin/disjointness, and write atomically."""
    if not input_paths:
        raise ValueError("input_paths must not be empty")
    if min_reward_margin < 0.0:
        raise ValueError("min_reward_margin must be >= 0.0")

    merged_rows: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    pairs_by_fold: Counter[str] = Counter()
    gold_chosen_count = 0
    sampled_chosen_count = 0
    rejected_loop_count = 0
    rejected_budget_capped_count = 0
    margins: list[float] = []

    for path in input_paths:
        rows = _read_jsonl(path)
        file_folds: set[str] = set()
        for idx, row in enumerate(rows):
            record_id = str(row.get("record_id") or "").strip()
            if not record_id:
                raise ValueError(f"{path}:{idx + 1} missing required non-empty record_id")
            if record_id in seen_record_ids:
                raise ValueError(f"Duplicate record_id across OOF preference shards: {record_id!r}")
            seen_record_ids.add(record_id)

            if not row.get("chosen") or not row.get("rejected"):
                raise ValueError(f"Record {record_id!r} in {path} is missing chosen or rejected turns")

            split_val = row.get("split")
            if split_val is not None and str(split_val).strip().lower() != "train":
                raise ValueError(f"Record {record_id!r} in {path} has non-train split={split_val!r}")

            source_fold = str(row.get("source_fold") or "").strip()
            if not source_fold or source_fold.lower() in _FORBIDDEN_FOLDS:
                raise ValueError(f"Record {record_id!r} in {path} has invalid source_fold={source_fold!r}")
            file_folds.add(source_fold)
            pairs_by_fold[source_fold] += 1

            margin = float(row.get("margin", 0.0))
            if margin < min_reward_margin:
                raise ValueError(f"Record {record_id!r} margin {margin:.6f} < min_reward_margin {min_reward_margin:.6f}")
            margins.append(margin)

            src = str(row.get("chosen_source") or "").strip().lower()
            if src == "gold":
                gold_chosen_count += 1
            elif src == "sampled":
                sampled_chosen_count += 1
            if row.get("rejected_is_loop"):
                rejected_loop_count += 1
            if row.get("rejected_is_budget_capped"):
                rejected_budget_capped_count += 1

            merged_rows.append(row)

        if require_distinct_folds and len(input_paths) > 1 and len(file_folds) > 1:
            raise ValueError(f"Expected single fold per shard when require_distinct_folds=True, got {sorted(file_folds)} in {path}")

    if not merged_rows:
        raise ValueError("No preference pairs found across input files")

    eval_record_ids: set[str] = set()
    eval_doc_ids: set[str] = set()
    eval_sha256s: set[str] = set()
    for eval_path in eval_manifests:
        for row in _read_jsonl(eval_path):
            rid = str(row.get("record_id") or "").strip()
            did = str(row.get("document_id") or "").strip()
            sha = str(row.get("image_sha256") or "").strip()
            if rid:
                eval_record_ids.add(rid)
            if did:
                eval_doc_ids.add(did)
            if sha:
                eval_sha256s.add(sha)

    col_rid = [r["record_id"] for r in merged_rows if str(r.get("record_id") or "").strip() in eval_record_ids]
    col_did = [r.get("document_id") for r in merged_rows if str(r.get("document_id") or "").strip() in eval_doc_ids]
    col_sha = [r.get("image_sha256") for r in merged_rows if str(r.get("image_sha256") or "").strip() in eval_sha256s]
    if col_rid or col_did or col_sha:
        raise AssertionError(
            f"OOF preference leakage vs eval manifests detected: record_id={len(col_rid)}, document_id={len(col_did)}, image_sha256={len(col_sha)}"
        )

    write_jsonl_atomic(out_jsonl, merged_rows)
    return {
        "total_pairs": len(merged_rows),
        "input_files": [str(p) for p in input_paths],
        "out_jsonl": str(out_jsonl),
        "pairs_by_fold": dict(sorted(pairs_by_fold.items())),
        "gold_chosen_count": gold_chosen_count,
        "sampled_chosen_count": sampled_chosen_count,
        "rejected_loop_count": rejected_loop_count,
        "rejected_budget_capped_count": rejected_budget_capped_count,
        "mean_margin": round(sum(margins) / len(margins), 6),
        "min_margin": round(min(margins), 6),
        "max_margin": round(max(margins), 6),
        "eval_leakage_audit": {
            "passed": True,
            "eval_manifests_checked": len(eval_manifests),
            "collisions": {"record_id": 0, "document_id": 0, "image_sha256": 0},
        },
    }
