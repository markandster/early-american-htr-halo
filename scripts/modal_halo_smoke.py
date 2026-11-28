"""Serverless 80 GB+ BF16 Halo smoke runner on Modal (runs scripts/run_halo_smoke.sh once and exits)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import modal

REPO_URL = "https://github.com/markandster/early-american-htr-halo.git"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "ca-certificates")
    .pip_install("uv")
)

app = modal.App("eahtr-halo-bf16-smoke")


@app.function(image=image, gpu="A100-80GB", timeout=3600)
def run_bf16_smoke(git_ref: str = "main", steps: int = 5) -> dict[str, object]:
    """Clone clean-room repo at ``git_ref``, verify GPU/VRAM, and run 5-step BF16 Halo smoke."""
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"[MODAL_GPU] {gpu_info}", flush=True)

    repo_dir = Path("/tmp/early-american-htr-halo")
    if repo_dir.exists():
        subprocess.run(["rm", "-rf", str(repo_dir)], check=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", git_ref, REPO_URL, str(repo_dir)], check=True)

    env = dict(os.environ)
    env.update(
        {
            "HALO_DIR": "/tmp/halo",
            "SMOKE_VENV": "/tmp/halo-gpu-smoke-venv",
            "WORK_DIR": "/tmp/eahtr_halo_smoke",
            "HF_HOME": "/tmp/hf_cache",
            "UV_CACHE_DIR": "/tmp/uv_cache",
            "STEPS": str(steps),
            "SKIP_SLOW_GEN": "0",
        }
    )
    proc = subprocess.run(["bash", "scripts/run_halo_smoke.sh"], cwd=repo_dir, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"run_halo_smoke.sh exited with code {proc.returncode}")
    return {"git_ref": git_ref, "steps": steps, "gpu": gpu_info, "status": "passed"}


@app.local_entrypoint()
def main(git_ref: str = "main", gpu: str = "A100-80GB", steps: int = 5) -> None:
    """Launch remote 80 GB+ BF16 Halo smoke (`--gpu A100-80GB` or `--gpu H100`)."""
    result = run_bf16_smoke.with_options(gpu=gpu).remote(git_ref=git_ref, steps=steps)
    print(result)
