"""Competition Halo Arrow dataset export and exported/MLX directory verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .halo_smoke import DEFAULT_SMOKE_PROMPT, filter_vlm_rows_post_expansion
from .hygiene import audit_split_disjointness


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


def export_halo_vlm_sft_dataset(
    train_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    out_dir: Path,
    count_tokens_fn: Callable[[str], int],
    *,
    max_length: int = 4096,
    vision_tokens: int = 1120,
    prompt_text: str = DEFAULT_SMOKE_PROMPT,
) -> dict[str, Any]:
    """Audit disjointness, filter post-expansion lengths, and export Halo SFT Arrow DatasetDict."""
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence as HFSeq, Value

    audit = audit_split_disjointness(list(train_records), list(test_records))
    if not audit["passed"]:
        raise ValueError(f"Split disjointness check failed before Halo SFT export: {audit}")

    kept_train, drop_train = filter_vlm_rows_post_expansion(
        train_records, count_tokens_fn, max_length=max_length, vision_tokens=vision_tokens, prompt_text=prompt_text
    )
    kept_test, drop_test = filter_vlm_rows_post_expansion(
        test_records, count_tokens_fn, max_length=max_length, vision_tokens=vision_tokens, prompt_text=prompt_text
    )
    if not kept_train or not kept_test:
        raise ValueError(
            f"Post-expansion filtering left empty split (train={len(kept_train)}, test={len(kept_test)})"
        )

    msg_feature = [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}]
    features = Features({"messages": msg_feature, "images": HFSeq(HFImage())})

    def _to_ds(rows: Sequence[dict[str, Any]]) -> Dataset:
        return Dataset.from_dict(
            {
                "messages": [
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image", "text": None}, {"type": "text", "text": r["prompt"]}],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": r["text"]}]},
                    ]
                    for r in rows
                ],
                "images": [[str(r["image_path"])] for r in rows],
            },
            features=features,
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": _to_ds(kept_train), "test": _to_ds(kept_test)}).save_to_disk(str(out_dir))
    return {
        "out_dir": str(out_dir),
        "train_rows": len(kept_train),
        "test_rows": len(kept_test),
        "dropped_train": drop_train,
        "dropped_test": drop_test,
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
) -> dict[str, Any]:
    """Filter preference pairs post-expansion and export Halo VLM SMPO/DPO Arrow DatasetDict."""
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence as HFSeq, Value

    def _filter_pairs(pairs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        kept: list[dict[str, Any]] = []
        dropped = 0
        max_completion = max_length - max_prompt_length
        for row in pairs:
            imgs = row.get("images") or ([row["image_path"]] if row.get("image_path") else [])
            if not imgs:
                dropped += 1
                continue
            prompt_norm = _normalize_msg_list(row.get("prompt"))
            chosen_norm = _normalize_msg_list(row.get("chosen"))
            rejected_norm = _normalize_msg_list(row.get("rejected"))
            p_toks = count_tokens_fn(_extract_text(prompt_norm))
            c_toks = count_tokens_fn(_extract_text(chosen_norm))
            r_toks = count_tokens_fn(_extract_text(rejected_norm))
            longest_comp = max(c_toks, r_toks)
            if (
                (vision_tokens + p_toks + 16) <= max_prompt_length
                and (longest_comp + 16) <= max_completion
                and (vision_tokens + p_toks + longest_comp + 32) <= max_length
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
            else:
                dropped += 1
        return kept, dropped

    kept_train, drop_train = _filter_pairs(train_pairs)
    kept_test, drop_test = _filter_pairs(test_pairs)
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

    def _to_ds(rows: Sequence[dict[str, Any]]) -> Dataset:
        payload: dict[str, Any] = {
            "prompt": [r["prompt"] for r in rows],
            "chosen": [r["chosen"] for r in rows],
            "rejected": [r["rejected"] for r in rows],
            "images": [r["images"] for r in rows],
        }
        if include_margin:
            payload["margin"] = [r["margin"] for r in rows]
        return Dataset.from_dict(payload, features=features)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": _to_ds(kept_train), "test": _to_ds(kept_test)}).save_to_disk(str(out_dir))
    return {
        "out_dir": str(out_dir),
        "train_rows": len(kept_train),
        "test_rows": len(kept_test),
        "dropped_train": drop_train,
        "dropped_test": drop_test,
        "include_margin": include_margin,
    }


def verify_exported_vlm_dir(
    model_dir: Path,
    *,
    expected_soft_tokens: int = 1120,
    check_gemma4_resolution: bool = True,
    check_nothink_template: bool = True,
) -> dict[str, Any]:
    """Fail closed if merged/MLX directory lost Gemma 4 resolution or non-thinking template."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Exported model directory not found: {model_dir}")

    checked_fields: dict[str, Any] = {"model_dir": str(model_dir)}
    if check_gemma4_resolution:
        proc_path = model_dir / "processor_config.json"
        if not proc_path.is_file():
            raise FileNotFoundError(f"Missing required {proc_path} in Gemma 4 export directory")
        data = json.loads(proc_path.read_text(encoding="utf-8"))
        if data.get("image_seq_length") != expected_soft_tokens:
            raise ValueError(
                f"{proc_path}: image_seq_length={data.get('image_seq_length')} != {expected_soft_tokens}"
            )
        ip = data.get("image_processor")
        if not isinstance(ip, dict) or ip.get("max_soft_tokens") != expected_soft_tokens:
            raise ValueError(
                f"{proc_path}: image_processor.max_soft_tokens={ip.get('max_soft_tokens') if isinstance(ip, dict) else None} "
                f"!= {expected_soft_tokens}"
            )
        if "image_seq_length" in ip and ip["image_seq_length"] != expected_soft_tokens:
            raise ValueError(
                f"{proc_path}: image_processor.image_seq_length={ip['image_seq_length']} != {expected_soft_tokens}"
            )
        checked_fields["image_seq_length"] = expected_soft_tokens

    if check_nothink_template:
        tpl_text = ""
        jinja_path = model_dir / "chat_template.jinja"
        tok_cfg_path = model_dir / "tokenizer_config.json"
        if jinja_path.is_file():
            tpl_text = jinja_path.read_text(encoding="utf-8")
        elif tok_cfg_path.is_file():
            tok_data = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
            raw = tok_data.get("chat_template")
            tpl_text = raw if isinstance(raw, str) else ""
        if not tpl_text.strip():
            raise ValueError(f"{model_dir}: missing chat_template.jinja and tokenizer_config.json chat_template")
        if "else 0.9" in tpl_text or "Thinking effort level: 0.9" in tpl_text:
            raise ValueError(f"{model_dir}: detected Inkling 0.9 reasoning-effort default instead of 0.0")
        checked_fields["chat_template_verified"] = True

    return checked_fields
