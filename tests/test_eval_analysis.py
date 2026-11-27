"""Tests for prediction error analysis, word-aligned line-break F1, and loop/cap buckets."""

from __future__ import annotations

import json
from pathlib import Path

from eahtr.eval_analysis import (
    aligned_line_boundary_f1,
    analyze_predictions,
    load_predictions_json,
)


def test_load_predictions_json_accepts_list_and_report_dict(tmp_path: Path) -> None:
    rows = [{"record_id": "p1", "prediction": "hello", "reference": "hello", "finish_reason": "stop"}]
    raw_file = tmp_path / "preds.json"
    raw_file.write_text(json.dumps(rows), encoding="utf-8")
    wrapped_file = tmp_path / "report.json"
    wrapped_file.write_text(json.dumps({"scores": {}, "predictions": rows}), encoding="utf-8")

    assert load_predictions_json(raw_file) == rows
    assert load_predictions_json(wrapped_file) == rows


def test_aligned_line_boundary_f1_diagnoses_word_offset_drift() -> None:
    ref = "line one words\nline two words\nline three words"
    # Inserting one word on line 1 shifts every cumulative word offset by +1.
    pred = "extra line one words\nline two words\nline three words"
    assert aligned_line_boundary_f1(pred, ref) == 1.0

    preds = [
        {
            "record_id": "doc1-p1",
            "document_id": "doc1",
            "prediction": pred + (" extra" * 30),
            "reference": ref + (" tail" * 30),
            "finish_reason": "stop",
        }
    ]
    # Test explicit offset drift counter on clean matched line count with shifted first word.
    drift_ref = "\n".join(f"alpha{i} beta{i} gamma{i} delta{i} epsilon{i}" for i in range(10))
    drift_pred = "zero " + drift_ref
    report = analyze_predictions(
        [{"record_id": "drift", "document_id": "d1", "prediction": drift_pred, "reference": drift_ref, "finish_reason": "stop"}]
    )
    assert report["line_break_diagnostics"]["raw_boundary_f1_pct"] == 0.0
    assert report["line_break_diagnostics"]["word_aligned_boundary_f1_pct"] == 100.0
    assert report["line_break_diagnostics"]["exact_line_count_pages"] == 1
    assert report["line_break_diagnostics"]["shifted_offset_pages"] == 1


def test_analyze_predictions_extracts_loops_openings_and_baseline_ci() -> None:
    loop_pred = "Dear Sir\n" + "\n".join(["same repeated line"] * 6)
    preds = [
        {
            "record_id": "r1",
            "document_id": "doc1",
            "prediction": "Clean line one\nClean line two",
            "reference": "Clean line one\nClean line two",
            "finish_reason": "stop",
        },
        {
            "record_id": "r2",
            "document_id": "doc2",
            "prediction": loop_pred,
            "reference": "Actual different opening\nSecond line text",
            "finish_reason": "length",
        },
    ]
    baseline = [
        {"record_id": "r1", "strict_cer": 0.9, "prediction": "garbled"},
        {"record_id": "r2", "strict_cer": 1.2, "prediction": "garbled"},
    ]
    manifest = [
        {"record_id": "r1", "document_id": "doc1", "is_two_page_opening": False, "text": "Clean line one\nClean line two"},
        {"record_id": "r2", "document_id": "doc2", "is_two_page_opening": True, "text": "Actual different opening\nSecond line text"},
    ]

    report = analyze_predictions(preds, baseline_predictions=baseline, manifest_rows=manifest, top_k=1)
    assert report["scores"]["loop_pages"] == 1
    assert report["scores"]["budget_capped_pages"] == 1
    assert len(report["loop_or_capped_pages"]) == 1
    assert report["loop_or_capped_pages"][0]["record_id"] == "r2"
    assert report["opening_breakdown"]["single_page"]["strict_cer_pct"] == 0.0
    assert report["opening_breakdown"]["two_page_opening"]["pages"] == 1
    assert report["worst_pages"][0]["record_id"] == "r2"
    assert report["baseline_comparison"]["matched_pages"] == 2
    assert report["baseline_comparison"]["mean_cer_diff_pp"] != 0.0
