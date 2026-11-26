"""Pre-competition 1x GPU Halo smoke harness utilities and notebook generator."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import yaml

from .candidates import DIPLOMATIC_PROMPT
from .hygiene import clean_diplomatic_target
from .revcity_range import fetch_revcity_smoke_sample

__all__ = ["fetch_revcity_smoke_sample"]

DEFAULT_SMOKE_PROMPT = DIPLOMATIC_PROMPT
_HALO_DOCKER_HINT = (
    "public.ecr.aws/whitecircle/halo:hopper-1.3.0rc2 or "
    "public.ecr.aws/whitecircle/halo:blackwell-1.3.0rc2"
)


def select_pytorch_cuda_index(driver_version: str) -> str:
    """Map NVIDIA driver version to the matching PyTorch 2.11.0 CUDA wheel index."""
    match = re.search(r"(\d+)\.(\d+)", (driver_version or "").strip())
    if not match:
        raise ValueError(f"Unrecognized NVIDIA driver version: {driver_version!r}")
    ver = (int(match.group(1)), int(match.group(2)))
    if ver >= (580, 65):
        return "https://download.pytorch.org/whl/cu130"
    if ver >= (570, 26):
        return "https://download.pytorch.org/whl/cu128"
    if ver >= (560, 28):
        return "https://download.pytorch.org/whl/cu126"
    raise ValueError(
        f"NVIDIA driver {driver_version!r} is older than 560.28 for PyTorch 2.11.0 wheels; "
        f"run via Halo Docker image ({_HALO_DOCKER_HINT}) instead."
    )


def prepare_gemma4_highres_dir(
    source_dir: Path,
    target_dir: Path,
    *,
    max_soft_tokens: int = 1120,
) -> Path:
    """Create a zero-copy symlink dir with patched Gemma 4 processor/preprocessor JSONs."""
    src = source_dir.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Gemma 4 cache directory not found: {src}")
    target_dir.mkdir(parents=True, exist_ok=True)

    json_targets = {"processor_config.json", "preprocessor_config.json"}
    for item in src.iterdir():
        dest = target_dir / item.name
        if item.name in json_targets and item.is_file():
            data = json.loads(item.read_text(encoding="utf-8"))
            if "max_soft_tokens" in data:
                data["max_soft_tokens"] = max_soft_tokens
            if "image_seq_length" in data:
                data["image_seq_length"] = max_soft_tokens
            ip = data.get("image_processor")
            if isinstance(ip, dict):
                if "max_soft_tokens" in ip:
                    ip["max_soft_tokens"] = max_soft_tokens
                if "image_seq_length" in ip:
                    ip["image_seq_length"] = max_soft_tokens
            dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        else:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            os.symlink(item, dest)
    return target_dir


def filter_vlm_rows_post_expansion(
    records: Sequence[dict[str, Any]],
    count_tokens_fn: Callable[[str], int],
    *,
    max_length: int,
    max_prompt_length: int | None = None,
    vision_tokens: int = 1120,
    prompt_text: str = DEFAULT_SMOKE_PROMPT,
) -> tuple[list[dict[str, Any]], int]:
    """Drop rows exceeding max_length after adding Gemma 4 expanded image tokens."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        row_prompt = str(rec.get("prompt") or prompt_text)
        target = clean_diplomatic_target(str(rec.get("text") or ""))
        if not target:
            dropped += 1
            continue
        p_toks = count_tokens_fn(row_prompt)
        t_toks = count_tokens_fn(target)
        # Include headroom for turn/control special tokens beyond raw text tokens.
        total_toks = vision_tokens + p_toks + t_toks + 32
        prompt_fit = max_prompt_length is None or (vision_tokens + p_toks + 16) <= max_prompt_length
        completion_fit = max_prompt_length is None or (t_toks + 16) <= (max_length - max_prompt_length)
        if total_toks <= max_length and prompt_fit and completion_fit:
            kept.append({**rec, "text": target, "prompt": row_prompt})
        else:
            dropped += 1
    return kept, dropped


def export_halo_vlm_smoke_datasets(
    records: Sequence[dict[str, Any]],
    out_root: Path,
    count_tokens_fn: Callable[[str], int],
    *,
    max_length: int,
    max_prompt_length: int,
    train_count: int = 12,
    test_count: int = 4,
    vision_tokens: int = 1120,
) -> tuple[Path, Path, int]:
    """Export Halo SFT and VLM SMPO Arrow datasets via save_to_disk with Sequence(Image())."""
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence, Value

    filtered, dropped = filter_vlm_rows_post_expansion(
        records,
        count_tokens_fn,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        vision_tokens=vision_tokens,
    )
    needed = train_count + test_count
    if len(filtered) < needed:
        raise ValueError(
            f"Only {len(filtered)} records fit post-expansion budget "
            f"(max_length={max_length}, max_prompt_length={max_prompt_length}); need {needed}."
        )
    selected = filtered[:needed]
    train_rows = selected[:train_count]
    test_rows = selected[train_count:needed]

    msg_feature = [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}]
    sft_features = Features({"messages": msg_feature, "images": Sequence(HFImage())})
    smpo_features = Features(
        {
            "prompt": msg_feature,
            "chosen": msg_feature,
            "rejected": msg_feature,
            "images": Sequence(HFImage()),
        }
    )

    def _sft_ds(rows: Sequence[dict[str, Any]]) -> Dataset:
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
            features=sft_features,
        )

    def _smpo_ds(rows: Sequence[dict[str, Any]]) -> Dataset:
        return Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": [{"type": "image", "text": None}, {"type": "text", "text": r["prompt"]}]}]
                    for r in rows
                ],
                "chosen": [[{"role": "assistant", "content": [{"type": "text", "text": r["text"]}]}] for r in rows],
                "rejected": [
                    [{"role": "assistant", "content": [{"type": "text", "text": f"{r['text']}\n-\n-\n-\n-"}]}]
                    for r in rows
                ],
                "images": [[str(r["image_path"])] for r in rows],
            },
            features=smpo_features,
        )

    sft_dir = out_root / "halo_vlm_sft_smoke"
    smpo_dir = out_root / "halo_vlm_smpo_smoke"
    DatasetDict({"train": _sft_ds(train_rows), "test": _sft_ds(test_rows)}).save_to_disk(str(sft_dir))
    DatasetDict({"train": _smpo_ds(train_rows), "test": _smpo_ds(test_rows)}).save_to_disk(str(smpo_dir))
    return sft_dir, smpo_dir, dropped


def build_smoke_yaml_configs(
    base_sft_yaml: Path,
    base_smpo_yaml: Path,
    out_dir: Path,
    *,
    model_dir: Path,
    smpo_model_dir: Path | None = None,
    sft_dataset_dir: Path,
    smpo_dataset_dir: Path,
    vram_gb: float,
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Emit temporary 1x GPU QLoRA SFT and SMPO smoke configs adapted for VRAM."""
    resolved_root = repo_root or base_sft_yaml.resolve().parents[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    small_vram = vram_gb < 32.0
    use_qlora = vram_gb < 56.0
    stage2_model = smpo_model_dir if smpo_model_dir is not None else out_dir / "sft_merged"

    def _patch(cfg: dict[str, Any], ds_path: Path, output_subdir: str, *, is_smpo: bool) -> dict[str, Any]:
        updated = dict(cfg)
        updated.pop("test_size", None)
        for fsdp_key in ("fsdp_reshard_after_forward", "fsdp_reshard_after_backward", "fsdp_defer_grad_sync"):
            updated.pop(fsdp_key, None)
        if isinstance(updated.get("chat_template"), str) and not Path(updated["chat_template"]).is_absolute():
            updated["chat_template"] = str((resolved_root / updated["chat_template"]).resolve())
        updated.update(
            {
                "model_name_or_path": str(stage2_model if is_smpo else model_dir),
                "bf16": True,
                "dataset": [str(ds_path)],
                "use_peft": True,
                "load_in_4bit": use_qlora,
                "use_bnb_nested_quant": use_qlora,
                "lora_target_modules": "all-linear",
                "gradient_checkpointing": True,
                "optim": "adamw_8bit" if use_qlora else "adamw_torch",
                "use_liger_kernel": False,
                "log_decoded_samples": True,
                "max_steps": 5,
                "save_strategy": "no" if is_smpo else "steps",
                "eval_strategy": "no",
                "report_to": "none",
                "output_dir": str(out_dir / output_subdir),
            }
        )
        if is_smpo:
            updated["chosen_sft_ratio"] = 1.0
            updated["max_length"] = 2048 if small_vram else 4096
            updated["max_prompt_length"] = 1350 if small_vram else 1600
        else:
            updated["save_steps"] = 5
            updated["max_length"] = 4096
        return updated

    sft_cfg = _patch(yaml.safe_load(base_sft_yaml.read_text(encoding="utf-8")), sft_dataset_dir, "sft_out", is_smpo=False)
    smpo_cfg = _patch(
        yaml.safe_load(base_smpo_yaml.read_text(encoding="utf-8")), smpo_dataset_dir, "smpo_out", is_smpo=True
    )
    sft_out = out_dir / "smoke_sft.yaml"
    smpo_out = out_dir / "smoke_vlm_smpo.yaml"
    sft_out.write_text(yaml.safe_dump(sft_cfg, sort_keys=False), encoding="utf-8")
    smpo_out.write_text(yaml.safe_dump(smpo_cfg, sort_keys=False), encoding="utf-8")
    return sft_out, smpo_out


def verify_processor_image_tokens(
    processor: Any,
    image_path: Path,
    *,
    prompt_text: str = DEFAULT_SMOKE_PROMPT,
    min_tokens: int = 281,
    max_tokens: int = 1120,
) -> int:
    """Encode a real manuscript page with processor and assert expanded image token count."""
    from PIL import Image

    if not image_path.is_file():
        raise FileNotFoundError(f"Sample image not found for processor verification: {image_path}")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    try:
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with Image.open(image_path) as raw_img:
        encoded = processor(text=[rendered], images=[raw_img.convert("RGB")], return_tensors="pt")

    ids = encoded["input_ids"][0]
    ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    tok = getattr(processor, "tokenizer", processor)
    img_tid = getattr(tok, "image_token_id", None)
    if img_tid is None:
        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
        img_tid = vocab.get("<|image|>") or vocab.get("<image_soft_token>")
    if img_tid is None:
        raise ValueError("Unable to resolve image_token_id on processor/tokenizer")

    count = sum(1 for tid in ids_list if int(tid) == int(img_tid))
    if not (min_tokens <= count <= max_tokens):
        raise ValueError(
            f"Realized processor image token count {count} outside [{min_tokens}, {max_tokens}]"
        )
    return count


_LOSS_LOG_RE = re.compile(r"['\"]?loss['\"]?\s*[:=]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)", re.I)


def verify_halo_run_log_finite_loss(log_path: Path, *, min_steps: int = 1) -> list[float]:
    """Assert Halo training run log recorded >= min_steps finite losses (at most 1 per line)."""
    if not log_path.is_file():
        raise FileNotFoundError(f"Expected Halo run log not found: {log_path}")
    losses: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LOSS_LOG_RE.search(line)
        if not match:
            continue
        val = float(match.group(1))
        if not math.isfinite(val):
            raise ValueError(f"Non-finite training loss {match.group(1)!r} detected in {log_path}")
        losses.append(val)
    if len(losses) < min_steps:
        raise ValueError(f"Found {len(losses)} training loss metrics (< {min_steps}) in {log_path}")
    return losses


def verify_stage_continuity_logprobs(
    base_lps: Sequence[float],
    sft_lps: Sequence[float],
    stage2_step0_lps: Sequence[float],
    *,
    tol: float = 1e-3,
    min_shift: float = 1e-4,
) -> dict[str, Any]:
    """Verify Stage 2 step-0 weights match Stage 1 merged SFT and diverge from base."""
    if not sft_lps or len(base_lps) != len(sft_lps) or len(stage2_step0_lps) != len(sft_lps):
        raise ValueError("Log-prob vectors must be non-empty and equal length")
    for seq in (base_lps, sft_lps, stage2_step0_lps):
        if any(not math.isfinite(float(v)) for v in seq):
            raise ValueError("Non-finite log-prob in stage continuity check")
    max_step0_diff = max(abs(float(a) - float(b)) for a, b in zip(sft_lps, stage2_step0_lps))
    max_sft_shift = max(abs(float(a) - float(b)) for a, b in zip(base_lps, sft_lps))
    if max_step0_diff > tol:
        raise ValueError(f"Stage 2 step-0 log-probs diverged from merged SFT (diff={max_step0_diff:.6f} > {tol})")
    if max_sft_shift < min_shift:
        raise ValueError(f"Merged SFT log-probs indistinguishable from base (shift={max_sft_shift:.6f} < {min_shift})")
    return {"max_step0_diff": max_step0_diff, "max_sft_shift": max_sft_shift, "n": len(sft_lps)}


def verify_train_sample_image_tokens(sample_txt: Path, *, min_tokens: int = 281, max_tokens: int = 1120) -> int:
    """Verify realized image placeholder tokens in Halo's logged train_sample.txt."""
    if not sample_txt.is_file():
        raise FileNotFoundError(f"Expected decoded sample log not found: {sample_txt}")
    text = sample_txt.read_text(encoding="utf-8")
    count = text.count("<|image|>") or text.count("<image_soft_token>")
    if not (min_tokens <= count <= max_tokens):
        raise ValueError(
            f"{sample_txt}: realized image token count {count} outside [{min_tokens}, {max_tokens}]"
        )
    return count


def write_halo_smoke_notebook(out_path: Path) -> Path:
    """Generate the pre-competition 1x GPU Halo SFT + SMPO smoke notebook."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Halo 1x GPU VLM Smoke Harness (SFT + SMPO)\n",
                "\n",
                "Runs an external `whitecircle/halo` Python 3.12 smoke check (5 steps of `sft.py` ",
                "and 5 steps of `smpo.py`) on 16 rights-cleared RevCity (`CC-BY-4.0`) manuscript pages, ",
                "verifying `1120` realized image tokens on the staged Gemma 4 processor and finite 5-step training loss.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import subprocess\n",
                "\n",
                "REPO_DIR = Path('/content/early-american-htr-halo')\n",
                "if not REPO_DIR.exists():\n",
                "    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/markandster/early-american-htr-halo.git', str(REPO_DIR)], check=True)\n",
                "else:\n",
                "    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=True)\n",
                "subprocess.run(['bash', str(REPO_DIR / 'scripts' / 'run_halo_smoke.sh')], cwd=str(REPO_DIR), check=True)",
            ],
        },
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return out_path
