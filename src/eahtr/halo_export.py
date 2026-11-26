"""Competition Halo Arrow dataset export and exported/MLX directory verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from .halo_export_verify import verify_exported_vlm_dir
from .halo_smoke import DEFAULT_SMOKE_PROMPT, filter_vlm_rows_post_expansion
from .hygiene import audit_split_disjointness
from .vision_budget import compute_target_grid_dimensions, encode_image_for_vlm


def _normalize_msg_list(messages: Any) -> list[dict[str, Any]]:
    norm: list[dict[str, Any]] = []
    for msg in messages or ():
        role = str(msg.get("role", ""))
        content = msg.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, (list, tuple)):
            parts = [
                {
                    "type": str(p.get("type", "text")),
                    "text": None if p.get("type") == "image" else str(p.get("text") or ""),
                }
                for p in content
                if isinstance(p, dict)
            ]
        else:
            parts = [{"type": "text", "text": str(content or "")}]
        norm.append({"role": role, "content": parts})
    return norm


def _extract_text(messages: Any) -> str:
    chunks: list[str] = []
    for msg in _normalize_msg_list(messages):
        for part in msg["content"]:
            if part.get("type") == "text" and part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


def _resolve_row_vision_tokens(
    image_path: str,
    max_image_tokens: int | None,
    cell_size: tuple[int, int],
    fallback_tokens: int,
) -> int:
    if max_image_tokens is None:
        return fallback_tokens
    from PIL import Image

    im = Image.open(str(image_path))
    try:
        w, h = im.size
    finally:
        if hasattr(im, "close"):
            im.close()
    _, _, tok_count = compute_target_grid_dimensions(
        w, h, max_image_tokens=max_image_tokens, cell_w=cell_size[0], cell_h=cell_size[1]
    )
    return tok_count


def _encode_image_entry(
    image_path: str,
    *,
    max_image_tokens: int | None,
    cell_size: tuple[int, int],
    image_format: str,
) -> Any:
    if max_image_tokens is None:
        return str(image_path)
    from PIL import Image

    im = Image.open(str(image_path))
    try:
        payload, _, _, _ = encode_image_for_vlm(
            im,
            max_image_tokens=max_image_tokens,
            cell_w=cell_size[0],
            cell_h=cell_size[1],
            image_format=image_format,
        )
    finally:
        if hasattr(im, "close"):
            im.close()
    return {"bytes": payload, "path": str(image_path)}


def _iter_sft_examples(
    rows: list[dict[str, Any]],
    max_image_tokens: int | None,
    cell_size: tuple[int, int],
    image_format: str,
) -> Any:
    for r in rows:
        yield {
            "messages": [
                {"role": "user", "content": [{"type": "image", "text": None}, {"type": "text", "text": r["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": r["text"]}]},
            ],
            "images": [
                _encode_image_entry(
                    str(r["image_path"]),
                    max_image_tokens=max_image_tokens,
                    cell_size=cell_size,
                    image_format=image_format,
                )
            ],
        }


def _iter_pref_examples(
    rows: list[dict[str, Any]],
    include_margin: bool,
    max_image_tokens: int | None,
    cell_size: tuple[int, int],
    image_format: str,
) -> Any:
    for r in rows:
        item: dict[str, Any] = {
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "images": [
                _encode_image_entry(
                    str(p),
                    max_image_tokens=max_image_tokens,
                    cell_size=cell_size,
                    image_format=image_format,
                )
                for p in r["images"]
            ],
        }
        if include_margin:
            item["margin"] = r["margin"]
        yield item


def export_halo_vlm_sft_dataset(
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    out_dir: Path,
    count_tokens_fn: Callable[[str], int],
    *,
    max_length: int = 4096,
    vision_tokens: int = 1120,
    prompt_text: str = DEFAULT_SMOKE_PROMPT,
    max_image_tokens: int | None = None,
    cell_size: tuple[int, int] = (40, 40),
    image_format: str = "PNG",
) -> dict[str, Any]:
    """Audit disjointness, filter post-expansion lengths, and stream-export Halo SFT Arrow DatasetDict."""
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence as HFSeq, Value

    audit = audit_split_disjointness(list(train_records), list(test_records))
    if not audit["passed"]:
        raise ValueError(f"Split disjointness check failed before Halo SFT export: {audit}")

    def _filter_sft(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        if max_image_tokens is None:
            kept, dropped = filter_vlm_rows_post_expansion(
                rows, count_tokens_fn, max_length=max_length, vision_tokens=vision_tokens, prompt_text=prompt_text
            )
            return kept, dropped, vision_tokens
        kept: list[dict[str, Any]] = []
        dropped = 0
        max_seen = 0
        for row in rows:
            prompt = row.get("prompt") or prompt_text
            row_vis = _resolve_row_vision_tokens(str(row["image_path"]), max_image_tokens, cell_size, vision_tokens)
            total = row_vis + count_tokens_fn(prompt) + count_tokens_fn(str(row["text"])) + 32
            if total <= max_length:
                kept.append({**row, "prompt": prompt})
                max_seen = max(max_seen, row_vis)
            else:
                dropped += 1
        return kept, dropped, max_seen

    kept_train, drop_train, max_vis_train = _filter_sft(train_records)
    kept_test, drop_test, max_vis_test = _filter_sft(test_records)
    if not kept_train or not kept_test:
        raise ValueError(
            f"Post-expansion filtering left empty split (train={len(kept_train)}, test={len(kept_test)})"
        )

    msg_feature = [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}]
    features = Features({"messages": msg_feature, "images": HFSeq(HFImage())})

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eahtr_hf_cache_") as cache_dir:
        ds_train = Dataset.from_generator(
            _iter_sft_examples,
            features=features,
            cache_dir=cache_dir,
            gen_kwargs={
                "rows": kept_train,
                "max_image_tokens": max_image_tokens,
                "cell_size": cell_size,
                "image_format": image_format,
            },
        )
        ds_test = Dataset.from_generator(
            _iter_sft_examples,
            features=features,
            cache_dir=cache_dir,
            gen_kwargs={
                "rows": kept_test,
                "max_image_tokens": max_image_tokens,
                "cell_size": cell_size,
                "image_format": image_format,
            },
        )
        DatasetDict({"train": ds_train, "test": ds_test}).save_to_disk(str(out_dir))
    return {
        "out_dir": str(out_dir),
        "train_rows": len(kept_train),
        "test_rows": len(kept_test),
        "dropped_train": drop_train,
        "dropped_test": drop_test,
        "max_vision_tokens": max(max_vis_train, max_vis_test),
    }


def export_halo_vlm_preference_dataset(
    train_pairs: Sequence[dict[str, Any]],
    test_pairs: Sequence[dict[str, Any]],
    out_dir: Path,
    count_tokens_fn: Callable[[str], int],
    *,
    max_length: int = 4096,
    max_prompt_length: int = 1600,
    vision_tokens: int = 1120,
    max_image_tokens: int | None = None,
    cell_size: tuple[int, int] = (40, 40),
    image_format: str = "PNG",
) -> dict[str, Any]:
    """Filter preference pairs post-expansion and stream-export Halo VLM SMPO Arrow DatasetDict."""
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence as HFSeq, Value

    def _filter_pairs(pairs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        kept: list[dict[str, Any]] = []
        dropped = 0
        max_seen = 0
        max_completion = max_length - max_prompt_length
        for row in pairs:
            imgs = row.get("images") or ([row["image_path"]] if row.get("image_path") else [])
            if not imgs:
                dropped += 1
                continue
            row_vis = sum(
                _resolve_row_vision_tokens(str(p), max_image_tokens, cell_size, vision_tokens)
                for p in imgs
            ) if max_image_tokens is not None else vision_tokens * len(imgs)
            prompt_norm = _normalize_msg_list(row.get("prompt"))
            chosen_norm = _normalize_msg_list(row.get("chosen"))
            rejected_norm = _normalize_msg_list(row.get("rejected"))
            p_toks = count_tokens_fn(_extract_text(prompt_norm))
            c_toks = count_tokens_fn(_extract_text(chosen_norm))
            r_toks = count_tokens_fn(_extract_text(rejected_norm))
            longest_comp = max(c_toks, r_toks)
            if (
                (row_vis + p_toks + 16) <= max_prompt_length
                and (longest_comp + 16) <= max_completion
                and (row_vis + p_toks + longest_comp + 32) <= max_length
            ):
                kept.append(
                    {
                        "prompt": prompt_norm,
                        "chosen": chosen_norm,
                        "rejected": rejected_norm,
                        "images": [str(p) for p in imgs],
                        "margin": float(row["margin"]) if row.get("margin") is not None else None,
                    }
                )
                max_seen = max(max_seen, row_vis)
            else:
                dropped += 1
        return kept, dropped, max_seen

    def _pair_audit_rows(pairs: Sequence[dict[str, Any]], *, forbid_eval_splits: bool = False) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(pairs):
            split_val = str(row.get("split") or "").strip().lower()
            if forbid_eval_splits and split_val != "train":
                raise ValueError(
                    f"Refusing non-train or missing split row ({split_val!r}) inside Stage-2 preference train_pairs"
                )
            imgs = row.get("images") or ([row["image_path"]] if row.get("image_path") else [])
            rows.append(
                {
                    "record_id": row.get("record_id", f"pair_{idx}"),
                    "document_id": row.get("document_id") or row.get("group_key") or "",
                    "writer_key": row.get("writer_key") or "",
                    "image_sha256": row.get("image_sha256") or "",
                    "image_dhash": row.get("image_dhash") or "",
                    "image_path": str(imgs[0]) if imgs else "",
                    "text": _extract_text(row.get("chosen")),
                }
            )
        return rows

    train_audit_rows = _pair_audit_rows(train_pairs, forbid_eval_splits=True)
    test_audit_rows = _pair_audit_rows(test_pairs)
    audit = audit_split_disjointness(train_audit_rows, test_audit_rows)
    path_collisions = sorted(
        {r["image_path"] for r in train_audit_rows if r["image_path"]}
        & {r["image_path"] for r in test_audit_rows if r["image_path"]}
    )
    if not audit["passed"] or path_collisions:
        audit["image_path_collisions"] = path_collisions
        raise ValueError(f"Split disjointness check failed before Halo preference export: {audit}")

    kept_train, drop_train, max_vis_train = _filter_pairs(train_pairs)
    kept_test, drop_test, max_vis_test = _filter_pairs(test_pairs)
    if not kept_train or not kept_test:
        raise ValueError(
            f"Post-expansion filtering left empty preference split (train={len(kept_train)}, test={len(kept_test)})"
        )

    include_margin = all(r["margin"] is not None for r in kept_train + kept_test)
    msg_feature = [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}]
    feat_dict: dict[str, Any] = {
        "prompt": msg_feature,
        "chosen": msg_feature,
        "rejected": msg_feature,
        "images": HFSeq(HFImage()),
    }
    if include_margin:
        feat_dict["margin"] = Value("float32")
    features = Features(feat_dict)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eahtr_hf_cache_") as cache_dir:
        ds_train = Dataset.from_generator(
            _iter_pref_examples,
            features=features,
            cache_dir=cache_dir,
            gen_kwargs={
                "rows": kept_train,
                "include_margin": include_margin,
                "max_image_tokens": max_image_tokens,
                "cell_size": cell_size,
                "image_format": image_format,
            },
        )
        ds_test = Dataset.from_generator(
            _iter_pref_examples,
            features=features,
            cache_dir=cache_dir,
            gen_kwargs={
                "rows": kept_test,
                "include_margin": include_margin,
                "max_image_tokens": max_image_tokens,
                "cell_size": cell_size,
                "image_format": image_format,
            },
        )
        DatasetDict({"train": ds_train, "test": ds_test}).save_to_disk(str(out_dir))
    return {
        "out_dir": str(out_dir),
        "train_rows": len(kept_train),
        "test_rows": len(kept_test),
        "dropped_train": drop_train,
        "dropped_test": drop_test,
        "include_margin": include_margin,
        "max_vision_tokens": max(max_vis_train, max_vis_test),
    }


__all__ = [
    "export_halo_vlm_preference_dataset",
    "export_halo_vlm_sft_dataset",
    "verify_exported_vlm_dir",
]
