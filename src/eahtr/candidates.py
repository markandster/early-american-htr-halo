"""Public-data-only Colab Pro training pipeline & notebook generator for Halo competition prep.

All datasets referenced here are strictly rights-cleared (CC-BY-4.0 or Public Domain),
and all candidate base models are Apache-2.0.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DIPLOMATIC_PROMPT = (
    "Transcribe the final manuscript text using only the visible ink, omitting "
    "crossed-out deletions and bracketed damage or editorial marks. Preserve the "
    "writer's original spelling, capitalization, punctuation, abbreviations, and "
    "line breaks. Do not modernize spelling or add commentary."
)

REVCITY_ZENODO_URL = "https://zenodo.org/api/records/15776323/files/revcity-htr.zip/content"
LOC_WASHINGTON_ITEM_JSON_URL = "https://www.loc.gov/item/2020446971/?fo=json"


@dataclass(frozen=True, slots=True)
class CompetitionCandidateSpec:
    """Configuration for a competition-eligible Apache-2.0 base VLM."""

    key: str
    hf_repo: str
    license: str
    params_b: float
    active_params_b: float
    lora_target_modules: tuple[str, ...]
    vision_trainable_patterns: tuple[str, ...]
    max_pixels: int
    mlx_quant_bits: int
    est_mlx_peak_gb: float
    min_bf16_vram_gb: float
    trust_remote_code: bool = False
    quant_method: str | None = None


COMPETITION_CANDIDATES: dict[str, CompetitionCandidateSpec] = {
    "gemma4_12b": CompetitionCandidateSpec(
        key="gemma4_12b",
        hf_repo="google/gemma-4-12B-it",
        license="Apache-2.0",
        params_b=12.0,
        active_params_b=12.0,
        lora_target_modules=("all-linear",),
        vision_trainable_patterns=(
            "vision_tower",
            "embed_vision",
            "multimodal_embedder",
            "patch_dense",
            "multi_modal_projector",
            "vision_projection",
        ),
        max_pixels=2048 * 2048,
        mlx_quant_bits=8,
        est_mlx_peak_gb=12.5,
        min_bf16_vram_gb=56.0,
        trust_remote_code=False,
    ),
    "gemma4_26b_a4b": CompetitionCandidateSpec(
        key="gemma4_26b_a4b",
        hf_repo="google/gemma-4-26B-A4B-it",
        license="Apache-2.0",
        params_b=26.0,
        active_params_b=4.0,
        lora_target_modules=("all-linear",),
        vision_trainable_patterns=(
            "vision_tower",
            "multi_modal_projector",
            "embed_vision",
            "vision_projection",
            "mm_projector",
            "merger",
        ),
        max_pixels=2048 * 2048,
        mlx_quant_bits=4,
        est_mlx_peak_gb=15.5,
        min_bf16_vram_gb=72.0,
        trust_remote_code=False,
    ),
    "inkling2_mini": CompetitionCandidateSpec(
        key="inkling2_mini",
        hf_repo="thinkingmachines/Inkling-2-Mini",
        license="Apache-2.0",
        params_b=26.4,
        active_params_b=5.4,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "r_proj", "o_proj"),
        vision_trainable_patterns=(
            "vision_projection",
            "multi_modal_projector",
            "patch_embed",
            "embed_vision",
            "visual.merger",
            "mm_projector",
            "merger",
        ),
        max_pixels=2048 * 2048,
        mlx_quant_bits=4,
        est_mlx_peak_gb=20.0,
        min_bf16_vram_gb=72.0,
        trust_remote_code=True,
        quant_method="modelopt",
    ),
}


def validate_competition_candidate(spec: CompetitionCandidateSpec) -> None:
    """Enforce Apache-2.0 license, full-res input, trainable vision, and <= 24 GB MLX bound."""
    if spec.license != "Apache-2.0":
        raise ValueError(f"Competition candidate {spec.hf_repo} must be Apache-2.0, got {spec.license}")
    if spec.est_mlx_peak_gb > 24.0:
        raise ValueError(f"Estimated MLX peak RAM exceeds 24 GB bound: {spec.est_mlx_peak_gb}")
    if spec.max_pixels < 1280 * 1280:
        raise ValueError(f"Full-resolution training requires max_pixels >= 1280*1280, got {spec.max_pixels}")
    if not spec.vision_trainable_patterns:
        raise ValueError(f"Vision projector/embedder patterns must be non-empty for {spec.hf_repo}")


def should_use_4bit(spec: CompetitionCandidateSpec, vram_gb: float, arm: str) -> bool:
    """Decide whether to load a model in 4-bit NF4 based on GPU VRAM and arm.

    Zero-shot inference needs roughly ``params_b * 2.5`` GB in BF16 (weights + KV/vision buffers),
    whereas LoRA fine-tuning requires ``spec.min_bf16_vram_gb`` to stay in full BF16 cleanly.
    """
    if arm == "baseline_zeroshot":
        return vram_gb < min(spec.min_bf16_vram_gb, spec.params_b * 2.5)
    return vram_gb < spec.min_bf16_vram_gb


def resolve_vision_modules_to_save(
    module_names: Iterable[str],
    patterns: Sequence[str],
) -> list[str]:
    """Select shallowest vision projector/embedder ancestors; fail closed if 0 match."""
    all_names = [n for n in module_names if n]
    matched: list[str] = []
    for name in all_names:
        if any(name == pat or name.endswith("." + pat) for pat in patterns):
            matched.append(name)

    deduped: list[str] = []
    for name in sorted(set(matched), key=lambda s: (s.count("."), s)):
        if not any(name.startswith(parent + ".") for parent in deduped):
            deduped.append(name)

    if not deduped:
        sample_top = sorted({n.split(".")[0] for n in all_names})[:15]
        raise RuntimeError(
            f"No trainable vision modules matched patterns {list(patterns)}. "
            f"Sample top-level modules: {sample_top}. Refusing frozen-vision run."
        )
    return deduped


def make_sham_derangement(records: Sequence[dict], seed: int = 20261125) -> list[dict]:
    """Deterministically mismatch images to transcripts with zero self-matches."""
    n = len(records)
    if n < 2:
        raise ValueError("Sham derangement requires at least 2 records")
    rng = random.Random(seed)
    indices = list(range(n))
    for _ in range(128):
        rng.shuffle(indices)
        if all(indices[i] != i for i in range(n)):
            break
    else:
        indices = [(i + 1) % n for i in range(n)]

    sham: list[dict] = []
    for i, rec in enumerate(records):
        donor = records[indices[i]]
        sham.append(
            {
                **rec,
                "text": donor["text"],
                "sham_donor_record_id": donor.get("record_id", str(indices[i])),
                "arm": "sham",
            }
        )
    return sham


def assign_deterministic_split(
    group_key: str,
    *,
    val_fraction: float = 0.075,
    test_fraction: float = 0.075,
    salt: str = "eahtr-public-split-v1",
) -> str:
    """Assign 'test_public', 'val', or 'train' deterministically by SHA-256 of group_key."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must lie in [0, 1)")
    digest = hashlib.sha256(f"{salt}:{group_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test_public"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


_REVCITY_STEM_RE = re.compile(r"^([A-Za-z]+)(?:%3A|_|:)(\d+)(?:_[A-Za-z0-9]+)?$")


def parse_revcity_stem_id(stem_name: str) -> tuple[str, int | None, str]:
    """Extract (institution, numeric_id, safe_id) from a RevCity filename stem."""
    base = Path(stem_name).name
    safe_id = base.replace("%3A", "_")
    match = _REVCITY_STEM_RE.match(base)
    if match:
        return match.group(1).lower(), int(match.group(2)), safe_id
    fallback_inst = safe_id.split("_", 1)[0].lower() or "revcity"
    return fallback_inst, None, safe_id


def assign_revcity_item_groups(
    stems: Sequence[str],
    *,
    max_id_gap: int = 1,
    creator_map: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Group RevCity pages by consecutive numeric ID runs per institution to keep multi-page items together.

    When archival creator metadata is supplied via ``creator_map``, uses attributed scribal hand
    as ``writer_key``; otherwise emits an unattributed item-scoped key so hand claims stay bounded.
    """
    parsed = [(stem, *parse_revcity_stem_id(stem)) for stem in stems]
    numeric_entries = sorted(
        [(inst, num, safe_id, stem) for stem, inst, num, safe_id in parsed if num is not None],
        key=lambda x: (x[0], x[1], x[2]),
    )
    groups: dict[str, dict[str, str]] = {}

    cur_inst = ""
    cur_start = -1
    cur_prev = -1
    run_items: list[tuple[str, str, str]] = []

    def _flush_run() -> None:
        if not run_items:
            return
        group_id = f"{cur_inst}:run_{cur_start}_{cur_prev}"
        for stem_val, inst_val, safe_val in run_items:
            creator = (creator_map or {}).get(safe_val) or (creator_map or {}).get(stem_val)
            writer_key = (
                f"scribed_hand:{creator.strip()}"
                if creator and str(creator).strip()
                else f"unattributed_item:{group_id}"
            )
            groups[stem_val] = {
                "institution": inst_val,
                "safe_id": safe_val,
                "document_id": group_id,
                "writer_key": writer_key,
                "hand_attributed": "true" if creator and str(creator).strip() else "false",
            }

    for inst, num, safe_id, stem in numeric_entries:
        if inst != cur_inst or num - cur_prev > max_id_gap:
            _flush_run()
            cur_inst = inst
            cur_start = num
            cur_prev = num
            run_items = [(stem, inst, safe_id)]
        else:
            cur_prev = num
            run_items.append((stem, inst, safe_id))
    _flush_run()

    for stem, inst, num, safe_id in parsed:
        if num is None:
            group_id = f"{inst}:stem_{safe_id}"
            creator = (creator_map or {}).get(safe_id) or (creator_map or {}).get(stem)
            writer_key = (
                f"scribed_hand:{creator.strip()}"
                if creator and str(creator).strip()
                else f"unattributed_item:{group_id}"
            )
            groups[stem] = {
                "institution": inst,
                "safe_id": safe_id,
                "document_id": group_id,
                "writer_key": writer_key,
                "hand_attributed": "true" if creator and str(creator).strip() else "false",
            }
    return groups
