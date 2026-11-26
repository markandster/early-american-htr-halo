"""Hardened preference pair construction for Stage 2 Halo VLM SMPO."""

from __future__ import annotations

from typing import Callable

from .candidates import DIPLOMATIC_PROMPT
from .metrics import normalize_preserve_newlines, normalize_strict
from .rewards import ArchivalRewardConfig, score_group_completions, score_transcription

DEFAULT_PREFERENCE_PROMPT = DIPLOMATIC_PROMPT


def _passes_nonblank_positive_gate(
    completion: str,
    reference: str,
    reward: float,
    *,
    min_chosen_reward: float,
    min_chosen_coverage: float,
) -> bool:
    ref_norm = normalize_strict(reference)
    pred_norm = normalize_strict(completion)
    if not ref_norm:
        return not pred_norm
    if not pred_norm or reward < min_chosen_reward:
        return False
    return (len(pred_norm) / len(ref_norm)) >= min_chosen_coverage


def build_preference_pairs(
    records_with_rollouts: list[dict[str, object]],
    min_reward_margin: float = 0.05,
    *,
    prompt_text: str = DEFAULT_PREFERENCE_PROMPT,
    config: ArchivalRewardConfig | None = None,
    variant_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
    prefer_gold_chosen: bool = True,
    min_chosen_reward: float = 0.50,
    min_chosen_coverage: float = 0.70,
    max_new_tokens_cap: int | None = None,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> list[dict[str, object]]:
    """Build Halo VLM SMPO preference pairs with coverage gating, gold anchor, and post-norm dedup."""
    if min_reward_margin < 0.0:
        raise ValueError("min_reward_margin must be non-negative")

    pairs: list[dict[str, object]] = []
    for rec in records_with_rollouts:
        image_path = str(rec.get("image_path") or "").strip()
        if not image_path:
            raise ValueError(f"Record {rec.get('record_id', '<unknown>')!r} is missing required image_path")

        completions_raw = rec.get("completions")
        completions = [str(c) for c in (completions_raw if isinstance(completions_raw, list) else [])]
        finish_reasons_raw = rec.get("finish_reasons")
        finish_reasons = (
            [str(fr) for fr in finish_reasons_raw] if isinstance(finish_reasons_raw, list) else None
        )
        reference = str(rec.get("text") or rec.get("reference") or "")
        scored = score_group_completions(
            completions,
            reference,
            finish_reasons,
            config=config,
            variant_pairs=variant_pairs,
        )
        if not scored:
            continue

        # Deduplicate rollouts after newline-preserving normalization, retaining highest-reward representative
        dedup_best_by_text: dict[str, dict[str, object]] = {}
        for item in scored:
            norm_key = normalize_preserve_newlines(str(item["completion"]))
            prev = dedup_best_by_text.get(norm_key)
            if prev is None or float(item["reward"]) > float(prev["reward"]):
                dedup_best_by_text[norm_key] = item
        unique_scored = list(dedup_best_by_text.values())

        sampled_sorted = sorted(unique_scored, key=lambda item: float(item["reward"]), reverse=True)
        qualifying_sampled = next(
            (
                item
                for item in sampled_sorted
                if _passes_nonblank_positive_gate(
                    str(item["completion"]),
                    reference,
                    float(item["reward"]),
                    min_chosen_reward=min_chosen_reward,
                    min_chosen_coverage=min_chosen_coverage,
                )
            ),
            None,
        )

        ref_norm = normalize_preserve_newlines(reference)
        chosen_text = ""
        chosen_reward = 0.0
        chosen_source = "sampled"
        if qualifying_sampled is not None and normalize_preserve_newlines(str(qualifying_sampled["completion"])) == ref_norm:
            chosen_text = str(qualifying_sampled["completion"])
            chosen_reward = float(qualifying_sampled["reward"])
        elif prefer_gold_chosen and ref_norm:
            chosen_text = reference
            chosen_reward = score_transcription(reference, reference, finish_reason="stop", config=config).reward
            chosen_source = "gold"
        elif qualifying_sampled is not None:
            chosen_text = str(qualifying_sampled["completion"])
            chosen_reward = float(qualifying_sampled["reward"])
        else:
            continue

        chosen_norm = normalize_preserve_newlines(chosen_text)
        neg_candidates = [
            item for item in unique_scored if normalize_preserve_newlines(str(item["completion"])) != chosen_norm
        ]
        if not neg_candidates:
            continue
        worst = min(neg_candidates, key=lambda item: float(item["reward"]))
        rejected_text = str(worst["completion"])

        rejected_reward = float(worst["reward"])
        margin = chosen_reward - rejected_reward
        if margin < min_reward_margin:
            continue

        rej_fr = str(worst.get("finish_reason") or "stop")
        rej_loop = bool(worst.get("loop_detected"))
        cap_val = int(max_new_tokens_cap or rec.get("max_new_tokens_cap") or 0)
        rej_toks = count_tokens_fn(rejected_text) if count_tokens_fn is not None else len(rejected_text.split())
        budget_censored = (rej_fr == "length") and not rej_loop

        row_prompt = str(rec.get("prompt") or prompt_text)
        pair: dict[str, object] = {
            "record_id": str(rec.get("record_id", "")),
            "images": [image_path],
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row_prompt}]}],
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
            "chosen_reward": round(chosen_reward, 6),
            "rejected_reward": round(rejected_reward, 6),
            "margin": round(margin, 6),
            "chosen_source": chosen_source,
            "rejected_finish_reason": rej_fr,
            "rejected_is_loop": rej_loop,
            "rejected_is_budget_capped": budget_censored,
            "max_new_tokens_cap": cap_val,
            "rejected_tokens": rej_toks,
        }
        for prov_key in ("document_id", "writer_key", "image_sha256", "image_dhash", "split", "source_checkpoint"):
            if rec.get(prov_key) is not None and str(rec.get(prov_key)).strip():
                pair[prov_key] = rec[prov_key]
        pairs.append(pair)
    return pairs
