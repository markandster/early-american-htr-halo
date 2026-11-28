"""Tests for hardened Hugging Face Hub checkpoint backup CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backup_checkpoint_hf as backup_cli


def _make_ckpt(root: Path, *, with_weights: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"architectures": ["Gemma4UnifiedForConditionalGeneration"]}', encoding="utf-8")
    (root / "._config.json").write_bytes(b"appledouble")
    (root / "optimizer.pt").write_bytes(b"opt" * 20)
    if with_weights:
        (root / "model.safetensors").write_bytes(b"weights" * 10)
    return root


def test_validate_and_dry_run_excludes_appledouble_and_optimizer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        backup_cli.validate_checkpoint_dir(tmp_path / "missing")

    no_cfg = tmp_path / "no_cfg"
    no_cfg.mkdir()
    with pytest.raises(ValueError, match="missing config.json"):
        backup_cli.validate_checkpoint_dir(no_cfg)

    no_wt = _make_ckpt(tmp_path / "no_wt", with_weights=False)
    with pytest.raises(ValueError, match="no \\*.safetensors"):
        backup_cli.validate_checkpoint_dir(no_wt)

    ckpt = _make_ckpt(tmp_path / "ok_ckpt")
    dry = backup_cli.upload_checkpoint_to_hf(ckpt, "owner/eahtr-test", dry_run=True)
    assert dry["status"] == "dry_run"
    assert dry["private"] is True
    assert dry["files"] == 2  # config.json + model.safetensors only

    proc = subprocess.run(
        [sys.executable, "scripts/backup_checkpoint_hf.py", "--checkpoint-dir", str(ckpt), "--repo-id", "owner/eahtr-test", "--dry-run"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "dry_run"


def test_upload_token_missing_secret_redaction_and_api_errors(tmp_path: Path, monkeypatch) -> None:
    ckpt = _make_ckpt(tmp_path / "ckpt")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    missing = backup_cli.upload_checkpoint_to_hf(ckpt, "owner/eahtr-test")
    assert missing["status"] == "error" and missing["reason"] == "missing_hf_token"

    calls: list[tuple[str, dict]] = []

    class FakeApi:
        def create_repo(self, **kw):
            calls.append(("create_repo", kw))

        def upload_folder(self, **kw):
            calls.append(("upload_folder", kw))

    secret = "hf_SECRET_TEST_TOKEN_DO_NOT_LEAK"
    monkeypatch.setenv("HF_TOKEN", secret)
    ok = backup_cli.upload_checkpoint_to_hf(ckpt, "owner/eahtr-test", path_in_repo="stage1", api=FakeApi())
    assert ok["status"] == "uploaded"
    assert ok["private"] is True
    assert secret not in json.dumps(ok)
    assert calls[0][1]["private"] is True and calls[1][1]["path_in_repo"] == "stage1"

    class BrokenApi:
        def create_repo(self, **kw):
            raise RuntimeError(f"401 Unauthorized for token {secret}")

    fail = backup_cli.upload_checkpoint_to_hf(ckpt, "owner/eahtr-test", api=BrokenApi())
    assert fail["status"] == "error" and fail["reason"] == "upload_failed"
    assert secret not in json.dumps(fail)
    assert "[REDACTED]" in fail["detail"]
