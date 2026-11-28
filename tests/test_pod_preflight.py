"""Tests for cloud pod pre-flight readiness checks and secret non-leakage."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from eahtr import pod_preflight


def test_nvidia_smi_variants(monkeypatch) -> None:
    ok_runner = lambda *a, **kw: SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3, 81559, 580.82.07\n", stderr="")
    res_ok = pod_preflight.check_nvidia_gpus(min_gpus=1, min_vram_gb=56.0, min_driver="560.28", runner=ok_runner)
    assert res_ok["status"] == "pass"
    assert res_ok["gpus"][0]["vram_gib"] >= 79.0

    low_vram = lambda *a, **kw: SimpleNamespace(returncode=0, stdout="NVIDIA L4, 23034, 580.82.07\n", stderr="")
    res_vram = pod_preflight.check_nvidia_gpus(min_vram_gb=56.0, runner=low_vram)
    assert res_vram["status"] == "fail" and res_vram["reason"] == "insufficient_vram"

    old_drv = lambda *a, **kw: SimpleNamespace(returncode=0, stdout="NVIDIA A100, 81920, 550.54.14\n", stderr="")
    res_drv = pod_preflight.check_nvidia_gpus(min_driver="560.28", runner=old_drv)
    assert res_drv["status"] == "fail" and res_drv["reason"] == "driver_too_old"

    missing = lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi"))
    res_miss = pod_preflight.check_nvidia_gpus(runner=missing)
    assert res_miss["status"] == "error" and res_miss["reason"] == "missing_binary"

    nonzero = lambda *a, **kw: SimpleNamespace(returncode=2, stdout="", stderr="driver error")
    res_nz = pod_preflight.check_nvidia_gpus(runner=nonzero)
    assert res_nz["status"] == "error" and res_nz["reason"] == "nonzero_exit"


def test_disk_packages_and_secret_redaction(tmp_path: Path, monkeypatch) -> None:
    disk_ok = pod_preflight.check_disk_space(tmp_path, min_free_gb=0.01)
    assert disk_ok["status"] == "pass"
    disk_fail = pod_preflight.check_disk_space(tmp_path, min_free_gb=1e9)
    assert disk_fail["status"] == "fail"

    versions = {"torch": "2.11.0", "transformers": "5.22.0", "trl": "1.14.2", "peft": "0.18.1", "datasets": "4.0.0", "Pillow": "12.0.0"}
    monkeypatch.setattr(pod_preflight.importlib_metadata, "version", lambda name: versions[name])
    assert pod_preflight.check_python_packages(strict=False)["status"] == "warn"
    assert pod_preflight.check_python_packages(strict=True)["status"] == "fail"

    secret = "hf_SUPERSECRET_TOKEN_VALUE_12345"
    monkeypatch.setenv("HF_TOKEN", secret)
    tok_res = pod_preflight.check_hf_token(required=True)
    assert tok_res["status"] == "pass" and tok_res["present"] is True
    assert secret not in json.dumps(tok_res)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert pod_preflight.check_hf_token(required=False)["status"] == "warn"
    assert pod_preflight.check_hf_token(required=True)["status"] == "fail"


def test_dataset_manifest_and_cli_exit(tmp_path: Path, monkeypatch) -> None:
    bad_ds = tmp_path / "broken_arrow"
    bad_ds.mkdir()
    assert pod_preflight.check_arrow_dataset_dir(bad_ds, load_rows=False)["status"] == "fail"

    ds_dir = tmp_path / "ok_arrow"
    (ds_dir / "train").mkdir(parents=True)
    (ds_dir / "test").mkdir(parents=True)
    (ds_dir / "dataset_dict.json").write_text('{"splits": ["train", "test"]}', encoding="utf-8")
    assert pod_preflight.check_arrow_dataset_dir(ds_dir, load_rows=False)["status"] == "pass"

    manifest = tmp_path / "val.jsonl"
    manifest.write_text(json.dumps({"record_id": "r1", "split": "val"}) + "\n", encoding="utf-8")
    m_res = pod_preflight.check_manifest_jsonl(manifest)
    assert m_res["status"] == "pass" and m_res["rows"] == 1

    monkeypatch.delenv("HF_TOKEN", raising=False)
    versions = {"torch": "2.11.0", "transformers": "5.23.0", "trl": "1.14.2", "peft": "0.18.1", "datasets": "4.0.0", "Pillow": "12.0.0"}
    monkeypatch.setattr(pod_preflight.importlib_metadata, "version", lambda name: versions[name])
    agg = pod_preflight.run_pod_preflight(
        work_dir=tmp_path, min_free_gb=0.01, skip_gpu=True,
        dataset_dirs=[ds_dir], manifests=[manifest], load_rows=False,
    )
    assert agg["ready"] is True
    assert "hf_token" in agg["warning_checks"]

    out_json = tmp_path / "preflight.json"
    proc = subprocess.run(
        [sys.executable, "scripts/preflight_pod.py", "--work-dir", str(tmp_path), "--min-disk-gb", "0.01", "--skip-gpu", "--skip-packages", "--dataset-dir", str(ds_dir), "--manifest", str(manifest), "--no-load-rows", "--out-json", str(out_json)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out_json.read_text(encoding="utf-8"))["ready"] is True
