#!/usr/bin/env python3
"""Back up merged BF16 checkpoints or PEFT adapters safely to a private Hugging Face Hub repo."""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import sys
from typing import Any

_ALWAYS_IGNORE_PATTERNS: tuple[str, ...] = ("._*", ".DS_Store", "__pycache__", "*.pyc")
_OPTIMIZER_IGNORE_PATTERNS: tuple[str, ...] = ("optimizer.pt", "optimizer.bin", "scheduler.pt", "rng_state*.pth")


def _should_ignore(rel_parts: tuple[str, ...], *, include_optimizer_state: bool) -> bool:
    patterns = _ALWAYS_IGNORE_PATTERNS if include_optimizer_state else (*_ALWAYS_IGNORE_PATTERNS, *_OPTIMIZER_IGNORE_PATTERNS)
    return any(any(fnmatch(part, pat) for pat in patterns) for part in rel_parts)


def validate_checkpoint_dir(ckpt_dir: Path, *, include_optimizer_state: bool = False) -> dict[str, Any]:
    """Verify checkpoint directory has config + model/adapter weights and compute upload payload size."""
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not ((ckpt_dir / "config.json").is_file() or (ckpt_dir / "adapter_config.json").is_file()):
        raise ValueError(f"Checkpoint directory {ckpt_dir} is missing config.json and adapter_config.json")

    file_count = 0
    total_bytes = 0
    has_weights = False
    for path in ckpt_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(ckpt_dir).parts
        if _should_ignore(rel_parts, include_optimizer_state=include_optimizer_state):
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        if path.suffix in {".safetensors", ".bin"}:
            has_weights = True

    if not has_weights:
        raise ValueError(f"Checkpoint directory {ckpt_dir} contains no *.safetensors or *.bin weight files")
    return {
        "checkpoint_dir": str(ckpt_dir),
        "files": file_count,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / float(1 << 30), 3),
    }


def upload_checkpoint_to_hf(
    ckpt_dir: Path,
    repo_id: str,
    *,
    path_in_repo: str = "",
    private: bool = True,
    commit_message: str | None = None,
    dry_run: bool = False,
    include_optimizer_state: bool = False,
    api: Any | None = None,
) -> dict[str, Any]:
    """Validate and upload a checkpoint directory to a Hugging Face Hub model repository."""
    if not repo_id.strip() or "/" not in repo_id:
        raise ValueError(f"Invalid Hugging Face repo_id (expected 'owner/name'): {repo_id!r}")

    info = validate_checkpoint_dir(ckpt_dir, include_optimizer_state=include_optimizer_state)
    ignore_patterns = list(
        _ALWAYS_IGNORE_PATTERNS if include_optimizer_state else (*_ALWAYS_IGNORE_PATTERNS, *_OPTIMIZER_IGNORE_PATTERNS)
    )
    if dry_run:
        return {
            "status": "dry_run",
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "private": private,
            "ignore_patterns": ignore_patterns,
            **info,
        }

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return {
            "status": "error",
            "reason": "missing_hf_token",
            "repo_id": repo_id,
            "detail": "HF_TOKEN environment variable is required to upload checkpoints to Hugging Face Hub",
        }

    try:
        client = api
        if client is None:
            from huggingface_hub import HfApi

            client = HfApi(token=token)
        client.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        client.upload_folder(
            folder_path=str(ckpt_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo or None,
            commit_message=commit_message or f"Back up {ckpt_dir.name}",
            ignore_patterns=ignore_patterns,
        )
    except Exception as exc:
        safe_detail = str(exc).replace(token, "[REDACTED]")
        return {
            "status": "error",
            "reason": "upload_failed",
            "repo_id": repo_id,
            "detail": f"Hugging Face upload failed ({type(exc).__name__}): {safe_detail}",
        }

    return {
        "status": "uploaded",
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "private": private,
        "url": f"https://huggingface.co/{repo_id}",
        **info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up a Halo checkpoint/adapter directory to Hugging Face Hub.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Target HF Hub repo, e.g. username/eahtr-gemma4-12b-sft")
    parser.add_argument("--path-in-repo", default="", help="Optional subfolder path inside the HF Hub repo")
    parser.add_argument("--public", action="store_true", help="Create repo as public (default is private)")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--include-optimizer-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        result = upload_checkpoint_to_hf(
            args.checkpoint_dir,
            args.repo_id,
            path_in_repo=args.path_in_repo,
            private=not args.public,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
            include_optimizer_state=args.include_optimizer_state,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "error", "reason": "validation_failed", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"uploaded", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
