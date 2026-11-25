"""Unit tests for verifiable archival GRPO rewards."""

import json
from pathlib import Path

from eahtr.rewards import (
    count_editorial_leaks,
    count_modernizations,
    detect_repetition_loop,
    score_group_completions,
    score_jsonl_file,
    score_transcription,
)


def test_detect_absorbing_dash_loop_after_wrap_hyphen() -> None:
    text = "you earnestly request my wri-\nting by every post\n" + ("-\n" * 655)
    looped, reasons = detect_repetition_loop(text)
    assert looped is True
    assert any("dash" in r for r in reasons)

    res = score_transcription(text, "you earnestly request my writing by every post and I remain yours")
    assert res.loop_detected is True
    assert res.reward == -1.0


def test_non_terminated_length_hits_hard_floor() -> None:
    res = score_transcription(
        "Boston July 6th 1776 Sir I have received",
        "Boston July 6th 1776 Sir I have received",
        finish_reason="length",
    )
    assert res.non_terminated is True
    assert res.reward == -1.0


def test_detect_ngram_cycle_loop() -> None:
    text = "Sir I have " + ("and the same " * 6)
    looped, reasons = detect_repetition_loop(text)
    assert looped is True
    assert "consecutive_ngram_loop" in reasons


def test_editorial_bracket_and_watermark_leaks() -> None:
    ref = "I am Sir your obedient servant Robt Harrison"
    pred = (
        "I am [Sir] your [obedient] servant Robt Harrison\n"
        "Image from the collections of the Society. Not to be reproduced without permission."
    )
    leaks = count_editorial_leaks(pred, ref)
    assert leaks == 4

    clean_res = score_transcription(ref, ref)
    leaky_res = score_transcription(pred, ref)
    assert leaky_res.editorial_leak_count == 4
    assert leaky_res.reward < clean_res.reward


def test_modernization_penalty_targets_archaic_forms() -> None:
    ref = "We chear the publick with usefull musick whilst encreased"
    preserved = "We chear the publick with usefull musick whilst encreased"
    modernized = "We cheer the public with useful music while increased"

    assert count_modernizations(preserved, ref) == 0
    assert count_modernizations(modernized, ref) == 6

    res_pres = score_transcription(preserved, ref)
    res_mod = score_transcription(modernized, ref)
    assert res_mod.modernizations == 6
    assert res_mod.reward < res_pres.reward


def test_blank_page_guard_and_truncation_penalty() -> None:
    blank_ok = score_transcription("", "")
    assert blank_ok.reward == 1.0
    assert blank_ok.blank_hallucination is False

    blank_bad = score_transcription("Boston 1776", "")
    assert blank_bad.blank_hallucination is True
    assert blank_bad.reward == -1.0

    ref = "Boston July 6 1776 I have the honour to inform you of our present situation in camp"
    short_pred = "Boston July 6 1776"
    trunc = score_transcription(short_pred, ref)
    assert trunc.truncation_triggered is True
    assert trunc.reward < 0.5


def test_group_scoring_and_jsonl_roundtrip(tmp_path: Path) -> None:
    ref = "Sir I remain your humble servant"
    completions = [
        "Sir I remain your humble servant",
        "Sir I remain your humblest",
        "Sir I remain\n-\n-\n-\n-\n-",
    ]
    rows = score_group_completions(completions, ref, ["stop", "stop", "length"])
    assert len(rows) == 3
    assert rows[0]["reward"] > rows[1]["reward"] > rows[2]["reward"]
    assert rows[0]["advantage"] > 0.0
    assert rows[2]["advantage"] < 0.0

    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    in_path.write_text(
        json.dumps({"prompt": "Transcribe", "reference": ref, "completions": completions}) + "\n",
        encoding="utf-8",
    )
    count = score_jsonl_file(in_path, out_path)
    assert count == 1
    written = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert len(written["scored_completions"]) == 3
