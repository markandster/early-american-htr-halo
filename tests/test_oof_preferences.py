"""Tests for multi-fold OOF preference merging and eval-leakage auditing."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from eahtr.oof_preferences import merge_and_audit_oof_preferences


def _make_pair(rid: str, did: str, sha: str, fold: str, *, margin: float = 0.25, source: str = "gold", split: str = "train", loop: bool = False) -> dict:
    return {
        "record_id": rid,
        "document_id": did,
        "image_sha256": sha,
        "split": split,
        "source_fold": fold,
        "chosen": [{"role": "assistant", "content": "Dear Sir"}],
        "rejected": [{"role": "assistant", "content": "Deer Sir"}],
        "chosen_reward": 0.9,
        "rejected_reward": round(0.9 - margin, 6),
        "margin": margin,
        "chosen_source": source,
        "rejected_is_loop": loop,
        "rejected_is_budget_capped": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_clean_multifold_merge_and_summary(tmp_path: Path) -> None:
    f0 = _write_jsonl(tmp_path / "f0.jsonl", [_make_pair("r1", "doc1", "sha1", "fold0", margin=0.30, loop=True)])
    f1 = _write_jsonl(tmp_path / "f1.jsonl", [_make_pair("r2", "doc2", "sha2", "fold1", margin=0.20, source="sampled")])
    val = _write_jsonl(tmp_path / "val.jsonl", [{"record_id": "v1", "document_id": "doc_val", "image_sha256": "sha_val", "split": "val"}])
    out = tmp_path / "merged.jsonl"

    summary = merge_and_audit_oof_preferences([f0, f1], out, eval_manifests=[val])
    assert summary["total_pairs"] == 2
    assert summary["pairs_by_fold"] == {"fold0": 1, "fold1": 1}
    assert summary["gold_chosen_count"] == 1
    assert summary["sampled_chosen_count"] == 1
    assert summary["rejected_loop_count"] == 1
    assert summary["eval_leakage_audit"]["passed"] is True
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_merge_rejects_duplicates_bad_folds_low_margin_and_eval_leakage(tmp_path: Path) -> None:
    f0 = _write_jsonl(tmp_path / "f0.jsonl", [_make_pair("r1", "doc1", "sha1", "fold0")])
    f_dup = _write_jsonl(tmp_path / "f_dup.jsonl", [_make_pair("r1", "doc2", "sha2", "fold1")])
    with pytest.raises(ValueError, match="Duplicate record_id"):
        merge_and_audit_oof_preferences([f0, f_dup], tmp_path / "out.jsonl")

    f_bad_fold = _write_jsonl(tmp_path / "f_val.jsonl", [_make_pair("r2", "doc2", "sha2", "val")])
    with pytest.raises(ValueError, match="invalid source_fold"):
        merge_and_audit_oof_preferences([f_bad_fold], tmp_path / "out.jsonl")

    f_low_margin = _write_jsonl(tmp_path / "f_low.jsonl", [_make_pair("r3", "doc3", "sha3", "fold2", margin=0.01)])
    with pytest.raises(ValueError, match="min_reward_margin"):
        merge_and_audit_oof_preferences([f_low_margin], tmp_path / "out.jsonl", min_reward_margin=0.05)

    for field, val_row in (
        ("record_id", {"record_id": "r1", "document_id": "docX", "image_sha256": "shaX", "split": "val"}),
        ("document_id", {"record_id": "vX", "document_id": "doc1", "image_sha256": "shaX", "split": "val"}),
        ("image_sha256", {"record_id": "vX", "document_id": "docX", "image_sha256": "sha1", "split": "test_public"}),
    ):
        eval_m = _write_jsonl(tmp_path / f"eval_{field}.jsonl", [val_row])
        with pytest.raises(AssertionError, match="leakage"):
            merge_and_audit_oof_preferences([f0], tmp_path / "out.jsonl", eval_manifests=[eval_m])
