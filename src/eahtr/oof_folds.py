"""Balanced group-disjoint out-of-fold manifest generator for Stage 2 rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hygiene import audit_split_disjointness
from .revcity_build import write_jsonl_atomic

_FORBIDDEN_NAME_PARTS = ("val", "test", "challenge", "hard_case")


def fold_key(row: Mapping[str, Any]) -> str:
    """Return writer_key when writer-attributed, otherwise document_id."""
    if row.get("writer_attributed"):
        return str(row.get("writer_key") or "")
    return str(row.get("document_id") or row.get("record_id") or "")


def assign_balanced_folds(
    rows: Sequence[Mapping[str, Any]],
    num_folds: int,
    *,
    seed: int = 20261125,
) -> dict[str, int]:
    """Assign each fold group greedily to the lightest-loaded fold by page count."""
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = fold_key(row)
        if not key:
            raise ValueError(f"Missing fold grouping key in row: {row.get('record_id')!r}")
        by_key.setdefault(key, []).append(row)
    if len(by_key) < num_folds:
        raise ValueError(f"Cannot build {num_folds} folds from only {len(by_key)} distinct groups")

    loads = [0] * num_folds
    fold_map: dict[str, int] = {}
    ordered = sorted(
        by_key,
        key=lambda k: (-len(by_key[k]), hashlib.sha256(f"{seed}:{k}".encode("utf-8")).hexdigest()),
    )
    for key in ordered:
        best_fold = min(range(num_folds), key=lambda f: loads[f])
        fold_map[key] = best_fold
        loads[best_fold] += len(by_key[key])
    return fold_map


def build_oof_fold_manifests(
    train_jsonl: Path,
    out_dir: Path,
    *,
    num_folds: int = 4,
    seed: int = 20261125,
) -> dict[str, Any]:
    """Split a train-only manifest into K disjoint out-of-fold train/holdout manifests."""
    lower_name = train_jsonl.name.lower()
    if any(part in lower_name for part in _FORBIDDEN_NAME_PARTS):
        raise ValueError(f"Refusing non-train manifest filename for OOF fold generation: {train_jsonl}")

    rows = [json.loads(ln) for ln in train_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        raise ValueError(f"Empty training manifest: {train_jsonl}")
    bad_splits = {str(r.get("split") or "") for r in rows} - {"train"}
    if bad_splits:
        raise ValueError(f"OOF fold generation requires split=='train' only, found {sorted(bad_splits)}")

    fold_map = assign_balanced_folds(rows, num_folds, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries: list[dict[str, Any]] = []
    for k in range(num_folds):
        fold_train = [r for r in rows if fold_map[fold_key(r)] != k]
        fold_holdout = [r for r in rows if fold_map[fold_key(r)] == k]
        audit = audit_split_disjointness(fold_train, fold_holdout, ngram_size=32, max_dhash_distance=4)
        if not audit["passed"]:
            raise AssertionError(f"OOF fold {k} leakage detected: {audit}")

        write_jsonl_atomic(out_dir / f"oof_fold{k}_train.jsonl", fold_train)
        write_jsonl_atomic(out_dir / f"oof_fold{k}_holdout.jsonl", fold_holdout)
        fold_summaries.append(
            {
                "fold": k,
                "train_pages": len(fold_train),
                "train_documents": len({r["document_id"] for r in fold_train}),
                "holdout_pages": len(fold_holdout),
                "holdout_documents": len({r["document_id"] for r in fold_holdout}),
                "dhash_review_count": int(audit["dhash_review_count"]),
                "passed": bool(audit["passed"]),
            }
        )

    summary = {
        "train_jsonl": str(train_jsonl),
        "out_dir": str(out_dir),
        "num_folds": num_folds,
        "seed": seed,
        "total_train_pages": len(rows),
        "folds": fold_summaries,
    }
    (out_dir / "oof_folds_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
