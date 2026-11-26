"""Unit tests for strict and lenient diplomatic WER/CER scoring."""

from eahtr.metrics import (
    normalize_lenient,
    normalize_preserve_newlines,
    normalize_strict,
    paired_bootstrap_ci,
    score_line_breaks,
    score_pair,
)


def test_normalize_strict_preserves_case_and_punctuation() -> None:
    raw = "  Camp   Middlebrook,\n  July 6th, 1779.  "
    assert normalize_strict(raw) == "Camp Middlebrook, July 6th, 1779."


def test_normalize_lenient_folds_case_and_strips_punctuation() -> None:
    raw = "I am, Sir, your's — very Hble Servt!"
    assert normalize_lenient(raw) == "i am sir your's very hble servt"


def test_score_pair_exact_match() -> None:
    res = score_pair("Dear Sir,\nI have the Honour", "Dear Sir, I have the Honour")
    assert res["strict"].wer == 0.0
    assert res["strict"].cer == 0.0
    assert res["lenient"].wer == 0.0
    assert res["lenient"].cer == 0.0


def test_score_pair_case_and_punct_only_affects_strict() -> None:
    pred = "Honourable Gentlemen, Boston July 6 1776"
    ref = "honourable gentlemen Boston July 6, 1776."
    res = score_pair(pred, ref)
    assert res["strict"].word_edits > 0
    assert res["lenient"].word_edits == 0
    assert res["lenient"].cer == 0.0


def test_score_pair_empty_reference_handling() -> None:
    empty_ok = score_pair("   \n ", "")
    assert empty_ok["strict"].wer == 0.0
    assert empty_ok["strict"].cer == 0.0

    hallucinated = score_pair("Some invented text", "")
    assert hallucinated["strict"].wer == 1.0
    assert hallucinated["strict"].cer == 1.0


def test_line_break_order_tolerant_and_bootstrap_ci() -> None:
    assert normalize_preserve_newlines("Line  one \r\n\nLine\t two") == "Line one\nLine two"
    exact = score_line_breaks("Line one\nLine two", "Line one\nLine two")
    expected_keys = {
        "newline_cer",
        "line_boundary_f1",
        "order_tolerant_cer",
        "newline_char_edits",
        "newline_ref_chars",
        "matched_boundaries",
        "pred_lines",
        "ref_lines",
        "order_tolerant_char_edits",
        "order_tolerant_ref_chars",
    }
    assert set(exact) == expected_keys
    assert exact["newline_cer"] == 0.0
    assert exact["line_boundary_f1"] == 1.0
    assert exact["matched_boundaries"] == 1

    short = score_line_breaks("A\nB", "X\nB")
    long_p = score_line_breaks("Dear Sir I remain\nyours", "Dear Sir I remain\nyours")
    micro_nl = (short["newline_char_edits"] + long_p["newline_char_edits"]) / (
        short["newline_ref_chars"] + long_p["newline_ref_chars"]
    )
    assert abs(micro_nl - (short["newline_cer"] + long_p["newline_cer"]) / 2.0) > 0.05

    reordered = score_line_breaks("Second block\nFirst block", "First block\nSecond block")
    assert reordered["order_tolerant_cer"] == 0.0
    assert reordered["newline_cer"] > 0.0

    merged_breaks = score_line_breaks("First block Second block", "First block\nSecond block")
    assert merged_breaks["order_tolerant_cer"] == 0.0
    assert merged_breaks["newline_cer"] > 0.0

    single_line = score_line_breaks("Only one line", "Only one line")
    assert single_line["line_boundary_f1"] == 1.0

    ci = paired_bootstrap_ci(
        [-0.05, -0.04, -0.06, -0.03],
        ["doc1", "doc1", "doc2", "doc3"],
        weights=[100, 200, 50, 50],
        n_boot=300,
        seed=7,
    )
    assert ci["mean"] < 0.0
    assert ci["ci_high"] < 0.0
    assert ci["n_clusters"] == 3.0
