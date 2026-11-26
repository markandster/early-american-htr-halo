"""Deterministic verifiable GRPO reward scoring for diplomatic 18th-century HTR."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .candidates import DIPLOMATIC_PROMPT
from .metrics import normalize_strict, score_pair

_BRACKET_SPAN_RE = re.compile(r"\[[^\]]+\]")
_CODE_FENCE_RE = re.compile(r"```")
_WATERMARK_PHRASES = (
    "not to be reproduced without permission",
    "image from the collections of",
)

DEFAULT_HISTORICAL_SPELLING_PAIRS: tuple[tuple[str, str], ...] = (
    ("chear", "cheer"),
    ("usefull", "useful"),
    ("dreadfull", "dreadful"),
    ("wonderfull", "wonderful"),
    ("publick", "public"),
    ("musick", "music"),
    ("encreased", "increased"),
    ("enform", "inform"),
    ("whilst", "while"),
    ("altho", "although"),
    ("acount", "account"),
    ("goverment", "government"),
)

_DEFAULT_SPELLING_REGEXES: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = tuple(
    (
        re.compile(rf"\b{re.escape(archaic.lower())}\b"),
        re.compile(rf"\b{re.escape(modern.lower())}\b"),
    )
    for archaic, modern in DEFAULT_HISTORICAL_SPELLING_PAIRS
)

DEFAULT_PREFERENCE_PROMPT = DIPLOMATIC_PROMPT


@dataclass(frozen=True, slots=True)
class ArchivalRewardConfig:
    """Weights and thresholds for verifiable diplomatic HTR rewards."""

    cer_weight: float = 0.6
    wer_weight: float = 0.4
    modernization_penalty: float = 0.05
    format_penalty: float = 0.10
    truncation_penalty: float = 0.25
    blank_hallucination_penalty: float = 1.0
    loop_floor: float = -1.0
    max_consecutive_repeat_lines: int = 4
    max_consecutive_dash_tokens: int = 4
    max_consecutive_ngram_repeats: int = 5
    min_length_ratio: float = 0.70


@dataclass(frozen=True, slots=True)
class ArchivalRewardResult:
    """Detailed verifiable reward breakdown for a single completion."""

    reward: float
    cer_strict: float
    wer_strict: float
    loop_detected: bool
    non_terminated: bool
    modernizations: int
    editorial_leak_count: int
    truncation_triggered: bool
    blank_hallucination: bool
    reasons: list[str] = field(default_factory=list)


def _repetition_streaks(text: str) -> tuple[int, int, int, int, str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    max_line = 1 if lines else 0
    best_line = lines[0] if lines else ""
    max_dash_line = 1 if (lines and lines[0] in {"-", "--", "—"}) else 0
    if lines:
        streak = 1
        d_streak = max_dash_line
        for idx in range(1, len(lines)):
            if lines[idx] == lines[idx - 1]:
                streak += 1
                if streak > max_line:
                    max_line = streak
                    best_line = lines[idx]
            else:
                streak = 1
            if lines[idx] in {"-", "--", "—"}:
                d_streak += 1
                max_dash_line = max(max_dash_line, d_streak)
            else:
                d_streak = 0

    tokens = (text or "").split()
    max_dash_tok = 0
    d_tok_streak = 0
    for tok in tokens:
        if tok in {"-", "--", "—"}:
            d_tok_streak += 1
            max_dash_tok = max(max_dash_tok, d_tok_streak)
        else:
            d_tok_streak = 0

    max_ngram = 1 if tokens else 0
    for ngram_len in (4, 3, 2):
        if len(tokens) < ngram_len * 2:
            continue
        for start in range(len(tokens) - ngram_len * 2 + 1):
            unit = tokens[start : start + ngram_len]
            reps = 1
            pos = start + ngram_len
            while pos + ngram_len <= len(tokens) and tokens[pos : pos + ngram_len] == unit:
                reps += 1
                pos += ngram_len
            max_ngram = max(max_ngram, reps)
    return max_line, max_dash_line, max_dash_tok, max_ngram, best_line


def detect_repetition_loop(
    text: str,
    max_repeat_lines: int = 4,
    max_repeat_dashes: int = 4,
    max_repeat_4grams: int = 5,
    *,
    reference: str = "",
) -> tuple[bool, list[str]]:
    """Detect loops exceeding thresholds and any genuine repetitions present in reference."""
    p_line, p_dline, p_dtok, p_ngram, best_line = _repetition_streaks(text)
    if reference:
        r_line, r_dline, r_dtok, r_ngram, _ = _repetition_streaks(reference)
    else:
        r_line = r_dline = r_dtok = r_ngram = 0

    reasons: list[str] = []
    if p_line >= max_repeat_lines and p_line > r_line:
        reasons.append(f"consecutive_line_loop:{best_line[:24]}")
    if p_dline >= max_repeat_dashes and p_dline > r_dline:
        reasons.append("consecutive_dash_loop")
    if p_dtok >= max_repeat_dashes and p_dtok > r_dtok:
        reasons.append("consecutive_dash_token_loop")
    if p_ngram >= max_repeat_4grams and p_ngram > r_ngram:
        reasons.append("consecutive_ngram_loop")
    return bool(reasons), reasons


def recommended_max_new_tokens(ref_tokens: int, *, min_tokens: int = 2048, multiplier: float = 1.5) -> int:
    """Scale decoding token budget above longest reference pages so faithful long transcriptions never cap out."""
    return max(int(min_tokens), math.ceil(float(ref_tokens) * float(multiplier)))


def count_editorial_leaks(prediction: str, reference: str) -> int:
    """Count unprompted bracket spans, code fences, and boilerplate watermarks beyond reference."""
    pred = prediction or ""
    ref = reference or ""

    bracket_excess = max(0, len(_BRACKET_SPAN_RE.findall(pred)) - len(_BRACKET_SPAN_RE.findall(ref)))
    fence_excess = max(0, len(_CODE_FENCE_RE.findall(pred)) - len(_CODE_FENCE_RE.findall(ref)))

    pred_low = pred.lower()
    ref_low = ref.lower()
    watermark_excess = sum(
        1 for phrase in _WATERMARK_PHRASES if phrase in pred_low and phrase not in ref_low
    )
    return bracket_excess + fence_excess + watermark_excess


def count_modernizations(
    prediction: str,
    reference: str,
    variant_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
) -> int:
    """Count archaic spellings present in reference that were replaced by modern forms in prediction."""
    if variant_pairs is None:
        compiled_pairs: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] | list[
            tuple[re.Pattern[str], re.Pattern[str]]
        ] = _DEFAULT_SPELLING_REGEXES
    else:
        compiled_pairs = [
            (
                re.compile(rf"\b{re.escape(archaic.lower())}\b"),
                re.compile(rf"\b{re.escape(modern.lower())}\b"),
            )
            for archaic, modern in variant_pairs
        ]

    pred_low = (prediction or "").lower()
    ref_low = (reference or "").lower()

    modernizations = 0
    for archaic_re, modern_re in compiled_pairs:
        ref_count = len(archaic_re.findall(ref_low))
        if ref_count == 0:
            continue
        pred_archaic = len(archaic_re.findall(pred_low))
        pred_modern = len(modern_re.findall(pred_low))
        ref_modern = len(modern_re.findall(ref_low))
        # Count conversions where archaic occurrences dropped and modern occurrences rose
        dropped = max(0, ref_count - pred_archaic)
        added_modern = max(0, pred_modern - ref_modern)
        modernizations += min(dropped, added_modern)
    return modernizations


def score_transcription(
    prediction: str,
    reference: str,
    *,
    finish_reason: str = "stop",
    config: ArchivalRewardConfig | None = None,
    variant_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
) -> ArchivalRewardResult:
    """Compute a deterministic verifiable reward in [-1.0, 1.0] for one HTR completion."""
    cfg = config or ArchivalRewardConfig()
    reasons: list[str] = []

    non_terminated = finish_reason == "length"
    if non_terminated:
        reasons.append("non_terminated_length")

    loop_detected, loop_reasons = detect_repetition_loop(
        prediction,
        max_repeat_lines=cfg.max_consecutive_repeat_lines,
        max_repeat_dashes=cfg.max_consecutive_dash_tokens,
        max_repeat_4grams=cfg.max_consecutive_ngram_repeats,
        reference=reference,
    )
    reasons.extend(loop_reasons)

    scores = score_pair(prediction, reference)
    strict = scores["strict"]
    ref_norm = normalize_strict(reference)
    pred_norm = normalize_strict(prediction)

    if non_terminated or loop_detected:
        return ArchivalRewardResult(
            reward=cfg.loop_floor,
            cer_strict=strict.cer,
            wer_strict=strict.wer,
            loop_detected=loop_detected,
            non_terminated=non_terminated,
            modernizations=count_modernizations(prediction, reference, variant_pairs),
            editorial_leak_count=count_editorial_leaks(prediction, reference),
            truncation_triggered=False,
            blank_hallucination=False,
            reasons=reasons,
        )

    if not ref_norm:
        blank_hallucination = bool(pred_norm)
        if blank_hallucination:
            reasons.append("blank_page_hallucination")
            reward = max(-1.0, 1.0 - cfg.blank_hallucination_penalty * 2.0)
        else:
            reward = 1.0
        return ArchivalRewardResult(
            reward=reward,
            cer_strict=strict.cer,
            wer_strict=strict.wer,
            loop_detected=False,
            non_terminated=False,
            modernizations=0,
            editorial_leak_count=count_editorial_leaks(prediction, reference),
            truncation_triggered=False,
            blank_hallucination=blank_hallucination,
            reasons=reasons,
        )

    cer_acc = max(0.0, 1.0 - strict.cer)
    wer_acc = max(0.0, 1.0 - strict.wer)
    base_reward = cfg.cer_weight * cer_acc + cfg.wer_weight * wer_acc

    modernizations = count_modernizations(prediction, reference, variant_pairs)
    editorial_leaks = count_editorial_leaks(prediction, reference)
    truncation_triggered = len(pred_norm) < cfg.min_length_ratio * len(ref_norm)

    if modernizations:
        reasons.append(f"modernizations:{modernizations}")
    if editorial_leaks:
        reasons.append(f"editorial_leaks:{editorial_leaks}")
    if truncation_triggered:
        reasons.append("truncation_under_min_ratio")

    total_penalty = (
        cfg.modernization_penalty * modernizations
        + cfg.format_penalty * editorial_leaks
        + (cfg.truncation_penalty if truncation_triggered else 0.0)
    )
    reward = max(-1.0, min(1.0, base_reward - total_penalty))

    return ArchivalRewardResult(
        reward=reward,
        cer_strict=strict.cer,
        wer_strict=strict.wer,
        loop_detected=False,
        non_terminated=False,
        modernizations=modernizations,
        editorial_leak_count=editorial_leaks,
        truncation_triggered=truncation_triggered,
        blank_hallucination=False,
        reasons=reasons,
    )


def score_group_completions(
    predictions: list[str],
    reference: str,
    finish_reasons: list[str] | None = None,
    *,
    config: ArchivalRewardConfig | None = None,
    variant_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
) -> list[dict[str, object]]:
    """Score a GRPO rollout group and compute group-relative standardized advantages."""
    reasons_in = finish_reasons if finish_reasons is not None else ["stop"] * len(predictions)
    if len(reasons_in) != len(predictions):
        raise ValueError("finish_reasons length must match predictions length")
    if not predictions:
        return []

    results = [
        score_transcription(
            pred,
            reference,
            finish_reason=fr,
            config=config,
            variant_pairs=variant_pairs,
        )
        for pred, fr in zip(predictions, reasons_in, strict=True)
    ]
    raw_rewards = [r.reward for r in results]
    mean_r = sum(raw_rewards) / len(raw_rewards)
    var_r = sum((r - mean_r) ** 2 for r in raw_rewards) / len(raw_rewards)
    std_r = math.sqrt(var_r)

    scored_rows: list[dict[str, object]] = []
    for pred, fr, res in zip(predictions, reasons_in, results, strict=True):
        advantage = 0.0 if std_r < 1e-8 else (res.reward - mean_r) / (std_r + 1e-8)
        scored_rows.append(
            {
                "completion": pred,
                "finish_reason": fr,
                "reward": res.reward,
                "advantage": advantage,
                "cer_strict": res.cer_strict,
                "wer_strict": res.wer_strict,
                "loop_detected": res.loop_detected,
                "non_terminated": res.non_terminated,
                "modernizations": res.modernizations,
                "editorial_leak_count": res.editorial_leak_count,
                "truncation_triggered": res.truncation_triggered,
                "blank_hallucination": res.blank_hallucination,
                "reasons": res.reasons,
            }
        )
    return scored_rows


def score_jsonl_file(input_path: Path, output_path: Path) -> int:
    """Score rollout groups from an input JSONL file and write scored JSONL for Offline GRPO."""
    rows_written = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            completions = row["completions"]
            reference = row["reference"]
            finish_reasons = row.get("finish_reasons")
            scored = score_group_completions(completions, reference, finish_reasons)
            out_row = {
                **row,
                "rewards": [float(item["reward"]) for item in scored],
                "scored_completions": scored,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            rows_written += 1
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Score HTR rollout groups for Halo Offline GRPO.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with completions and reference")
    parser.add_argument("--output", type=Path, required=True, help="Output scored JSONL path")
    args = parser.parse_args()
    count = score_jsonl_file(args.input, args.output)
    print(f"Scored {count} prompt groups -> {args.output}")


if __name__ == "__main__":
    main()


def __getattr__(name: str) -> object:
    if name == "build_preference_pairs":
        from .preferences import build_preference_pairs

        return build_preference_pairs
    raise AttributeError(name)
