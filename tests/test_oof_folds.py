"""Tests for balanced OOF fold manifests and matched-compute Halo sham export."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from eahtr.candidates import make_sham_derangement
from eahtr.oof_folds import assign_balanced_folds, build_oof_fold_manifests


def _sample_train_rows() -> list[dict]:
    rows: list[dict] = []
    sizes = [6, 5, 4, 4, 3, 3]
    for doc_idx, n_pages in enumerate(sizes):
        for p in range(n_pages):
            rid = f"doc{doc_idx}-p{p}"
            rows.append(
                {
                    "record_id": rid,
                    "document_id": f"doc:{doc_idx}",
                    "writer_key": f"doc:{doc_idx}",
                    "writer_attributed": False,
                    "image_sha256": f"sha-{rid}",
                    "image_path": f"../pages/{rid}.jpg",
                    "text": f"Unique manuscript line {rid} for testing disjointness with enough distinct words.",
                    "split": "train",
                }
            )
    return rows


def test_build_oof_fold_manifests_balances_groups_and_audits_disjointness(tmp_path: Path) -> None:
    rows = _sample_train_rows()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_path = manifest_dir / "train.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    fold_map = assign_balanced_folds(rows, 4, seed=42)
    assert len(set(fold_map.values())) == 4

    out_dir = manifest_dir / "oof"
    summary = build_oof_fold_manifests(train_path, out_dir, num_folds=4, seed=42)
    assert summary["total_train_pages"] == len(rows)
    assert len(summary["folds"]) == 4
    assert all(f["passed"] for f in summary["folds"])
    assert sum(f["holdout_pages"] for f in summary["folds"]) == len(rows)

    sample_oof_row = json.loads((out_dir / "oof_fold0_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sample_oof_row["image_path"].startswith("../../pages/")
    assert (out_dir / sample_oof_row["image_path"]).resolve().parent == tmp_path / "pages"

    bad_name = tmp_path / "val.jsonl"
    bad_name.write_text(train_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing non-train"):
        build_oof_fold_manifests(bad_name, out_dir, num_folds=4)

    bad_split = tmp_path / "mixed.jsonl"
    bad_split.write_text(json.dumps({**rows[0], "split": "val"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split=='train' only"):
        build_oof_fold_manifests(bad_split, out_dir, num_folds=2)

    with pytest.raises(ValueError, match="distinct groups"):
        build_oof_fold_manifests(train_path, out_dir, num_folds=10)


def test_sham_derangement_and_export_cli_preference_guard(tmp_path: Path) -> None:
    rows = _sample_train_rows()[:4]
    sham = make_sham_derangement(rows, seed=20261125)
    assert len(sham) == len(rows)
    assert all(s["sham_donor_record_id"] != s["record_id"] for s in sham)
    assert all(s["text"] != orig["text"] for s, orig in zip(sham, rows))

    pref_dummy = tmp_path / "pref.jsonl"
    pref_dummy.write_text("{}\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/export_halo_datasets.py",
            "--mode",
            "preference",
            "--sham",
            "--train-jsonl",
            str(pref_dummy),
            "--val-jsonl",
            str(pref_dummy),
            "--out-dir",
            str(tmp_path / "out"),
            "--tokenizer",
            "dummy",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "--sham is only supported in --mode sft" in proc.stderr
