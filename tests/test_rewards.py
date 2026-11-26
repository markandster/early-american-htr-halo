"""Unit tests for verifiable archival GRPO rewards."""

import json
from pathlib import Path
import time

import pytest

from eahtr.rewards import (
    build_preference_pairs,
    count_editorial_leaks,
    count_modernizations,
    detect_repetition_loop,
    recommended_max_new_tokens,
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
    assert len(written["rewards"]) == 3
    assert written["rewards"][0] > written["rewards"][1] > written["rewards"][2]


def test_build_preference_pairs_selection_filtering_and_error_guards() -> None:
    ref = "Head Quarters July 1779 Sir I remain yours"
    records = [
        {
            "record_id": "rec-1",
            "image_path": "/tmp/page1.jpg",
            "text": ref,
            "completions": [ref, "Head Quarters\n-\n-\n-\n-\n-"],
            "finish_reasons": ["stop", "length"],
        },
        {
            "record_id": "rec-tied",
            "image_path": "/tmp/page2.jpg",
            "text": ref,
            "completions": [ref, ref],
            "finish_reasons": ["stop", "stop"],
        },
        {
            "record_id": "rec-empty",
            "image_path": "/tmp/page3.jpg",
            "text": ref,
            "completions": [],
        },
    ]
    pairs = build_preference_pairs(records, min_reward_margin=0.05)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["images"] == ["/tmp/page1.jpg"]
    assert pair["prompt"][0]["role"] == "user"
    assert pair["prompt"][0]["content"][0] == {"type": "image"}
    assert pair["chosen"] == [{"role": "assistant", "content": ref}]
    assert pair["rejected_reward"] == -1.0
    assert pair["margin"] > 0.05

    assert build_preference_pairs(records[1:], min_reward_margin=0.05) == []

    with pytest.raises(ValueError, match="image_path"):
        build_preference_pairs([{"record_id": "no-img", "text": ref, "completions": [ref, "other"]}])

    assert detect_repetition_loop("Ditto\nDitto\nDitto\nDitto\nDitto", reference="Ditto\nDitto\nDitto\nDitto\nDitto")[0] is False
    assert recommended_max_new_tokens(1200) == 2048
    assert recommended_max_new_tokens(2000) == 3000

    with pytest.raises(ValueError, match="finish_reasons"):
        build_preference_pairs(
            [{"record_id": "bad-fr", "image_path": "/tmp/p.jpg", "text": ref, "completions": [ref, "x"], "finish_reasons": ["stop"]}]
        )

    blank_chosen_guard = build_preference_pairs(
        [{"record_id": "g1", "image_path": "/tmp/p1.jpg", "text": ref, "completions": ["   ", "Head Quarters\n-\n-\n-\n-\n-"], "finish_reasons": ["stop", "length"]}],
        min_reward_margin=0.05,
        prefer_gold_chosen=True,
    )
    assert len(blank_chosen_guard) == 1
    assert blank_chosen_guard[0]["chosen"] == [{"role": "assistant", "content": ref}]
    assert blank_chosen_guard[0]["chosen_source"] == "gold"
    assert blank_chosen_guard[0]["rejected_is_loop"] is True
    assert blank_chosen_guard[0]["rejected_is_budget_capped"] is False

    capped_rec = build_preference_pairs(
        [{
            "record_id": "cap1", "document_id": "doc-cap", "writer_key": "w-cap", "image_sha256": "sha-cap",
            "split": "train", "source_checkpoint": "ckpt-5", "image_path": "/tmp/p2.jpg",
            "text": ref, "max_new_tokens_cap": 64, "completions": [ref, "Head Quarters July"], "finish_reasons": ["stop", "length"],
        }],
        min_reward_margin=0.05,
    )
    assert len(capped_rec) == 1
    assert capped_rec[0]["rejected_is_loop"] is False
    assert capped_rec[0]["rejected_is_budget_capped"] is True
    assert capped_rec[0]["document_id"] == "doc-cap"
    assert capped_rec[0]["image_sha256"] == "sha-cap"
    assert capped_rec[0]["source_checkpoint"] == "ckpt-5"

    newline_dup = build_preference_pairs(
        [{"record_id": "nd", "image_path": "/tmp/p2.jpg", "text": "Sir\nI am yours", "completions": ["Sir\nI am yours", "Sir I am yours"], "finish_reasons": ["stop", "stop"]}],
        min_reward_margin=0.01,
    )
    assert newline_dup == []


def test_score_transcription_precompiled_regex_speed() -> None:
    ref = "We chear the publick with usefull musick whilst encreased"
    pred = "We cheer the public with useful music while increased"
    t0 = time.perf_counter()
    for _ in range(1000):
        res = score_transcription(pred, ref)
        assert res.modernizations == 6
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"Expected precompiled 1,000 calls < 0.5s, took {elapsed:.3f}s"
