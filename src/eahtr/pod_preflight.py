"""Turnkey Dec 1 cloud pod pre-flight readiness checks with explicit failure states."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .halo_validate import validate_yaml_configs_tier2

DEFAULT_PINNED_PACKAGES: dict[str, str] = {
    "torch": "2.11.0",
    "transformers": "5.23.0",
    "trl": "1.14.2",
    "peft": "0.18.1",
}
REQUIRED_RUNTIME_PACKAGES: tuple[str, ...] = ("datasets", "Pillow")


def parse_driver_version(version_str: str) -> tuple[int, ...]:
    """Parse a numeric NVIDIA driver version string into an integer tuple."""
    parts = [int(tok) for tok in re.findall(r"\d+", version_str.strip())]
    if not parts:
        raise ValueError(f"Unparseable NVIDIA driver version: {version_str!r}")
    return tuple(parts)


def check_nvidia_gpus(
    *,
    min_gpus: int = 1,
    min_vram_gb: float = 56.0,
    min_driver: str = "560.28",
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Verify visible NVIDIA GPUs, freeable VRAM capacity (>= GiB), and driver version."""
    exec_fn = runner or subprocess.run
    try:
        proc = exec_fn(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except FileNotFoundError:
        return {"name": "gpu", "status": "error", "reason": "missing_binary", "detail": "nvidia-smi not found on PATH", "gpus": []}
    except Exception as exc:
        return {"name": "gpu", "status": "error", "reason": "subprocess_error", "detail": f"nvidia-smi failed: {exc}", "gpus": []}

    if int(getattr(proc, "returncode", 1)) != 0:
        err = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        return {"name": "gpu", "status": "error", "reason": "nonzero_exit", "detail": f"nvidia-smi exited {proc.returncode}: {err}", "gpus": []}

    gpus: list[dict[str, Any]] = []
    min_drv_tuple = parse_driver_version(min_driver)
    for raw_line in (getattr(proc, "stdout", "") or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return {"name": "gpu", "status": "error", "reason": "parse_error", "detail": f"Unexpected nvidia-smi row: {line!r}", "gpus": []}
        try:
            vram_mib = float(parts[1])
            drv_tuple = parse_driver_version(parts[2])
        except ValueError as exc:
            return {"name": "gpu", "status": "error", "reason": "parse_error", "detail": str(exc), "gpus": []}
        gpus.append({"name": parts[0], "vram_gib": round(vram_mib / 1024.0, 2), "driver_version": parts[2], "driver_tuple": drv_tuple})

    if len(gpus) < min_gpus:
        return {"name": "gpu", "status": "fail", "reason": "insufficient_gpus", "detail": f"Found {len(gpus)} GPUs; need >= {min_gpus}", "gpus": []}

    old_drivers = [g for g in gpus if g["driver_tuple"] < min_drv_tuple]
    low_vram = [g for g in gpus if g["vram_gib"] < min_vram_gb]
    clean_gpus = [{k: v for k, v in g.items() if k != "driver_tuple"} for g in gpus]
    if old_drivers:
        return {"name": "gpu", "status": "fail", "reason": "driver_too_old", "detail": f"Driver below {min_driver} on {len(old_drivers)} GPU(s)", "gpus": clean_gpus}
    if low_vram:
        return {"name": "gpu", "status": "fail", "reason": "insufficient_vram", "detail": f"VRAM below {min_vram_gb} GiB on {len(low_vram)} GPU(s)", "gpus": clean_gpus}
    return {"name": "gpu", "status": "pass", "detail": f"{len(clean_gpus)} GPU(s) verified >= {min_vram_gb} GiB, driver >= {min_driver}", "gpus": clean_gpus}


def check_disk_space(path: Path, *, min_free_gb: float = 100.0) -> dict[str, Any]:
    """Check free disk space at ``path`` (walking up to nearest existing parent)."""
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {"name": f"disk:{path}", "status": "error", "detail": f"disk_usage failed: {exc}"}
    free_gib = round(usage.free / float(1 << 30), 2)
    status = "pass" if free_gib >= min_free_gb else "fail"
    return {"name": f"disk:{path}", "status": status, "free_gib": free_gib, "min_free_gib": min_free_gb, "detail": f"{free_gib} GiB free at {probe} (need >= {min_free_gb} GiB)"}


def check_python_packages(
    expected: Mapping[str, str] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Verify required ML/vision libraries and check pinned versions."""
    pinned = dict(expected if expected is not None else DEFAULT_PINNED_PACKAGES)
    installed: dict[str, str | None] = {}
    missing: list[str] = []
    mismatches: list[str] = []

    for pkg in (*pinned.keys(), *REQUIRED_RUNTIME_PACKAGES):
        try:
            ver = importlib_metadata.version(pkg)
            installed[pkg] = ver
        except importlib_metadata.PackageNotFoundError:
            installed[pkg] = None
            missing.append(pkg)
            continue
        if pkg in pinned and ver != pinned[pkg]:
            mismatches.append(f"{pkg}=={ver} (expected {pinned[pkg]})")

    if missing:
        return {"name": "packages", "status": "fail", "installed": installed, "detail": f"Missing package(s): {', '.join(missing)}"}
    if mismatches:
        status = "fail" if strict else "warn"
        return {"name": "packages", "status": status, "installed": installed, "detail": f"Version mismatch(es): {'; '.join(mismatches)}"}
    return {"name": "packages", "status": "pass", "installed": installed, "detail": "All pinned ML and runtime packages verified"}


def check_hf_token(*, required: bool = False) -> dict[str, Any]:
    """Check HF_TOKEN presence without ever logging or returning its value."""
    present = bool(os.environ.get("HF_TOKEN", "").strip())
    if present:
        return {"name": "hf_token", "status": "pass", "present": True, "detail": "HF_TOKEN is set in environment"}
    status = "fail" if required else "warn"
    return {"name": "hf_token", "status": status, "present": False, "detail": "HF_TOKEN unset (required only if uploading datasets or pulling gated assets)"}


def check_halo_repo(halo_repo: Path, config_dir: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Verify external whitecircle/halo layout and run Tier-1 + Tier-2 YAML validation."""
    required_files = (
        halo_repo / "scripts" / "training" / "sft.py",
        halo_repo / "scripts" / "training" / "preference" / "smpo.py",
        halo_repo / "scripts" / "after_training" / "merge_peft_adapters.py",
    )
    missing = [str(p) for p in required_files if not p.is_file()]
    if missing:
        return {"name": "halo_repo", "status": "fail", "detail": f"Missing Halo entrypoint(s): {', '.join(missing)}"}
    try:
        tier2 = validate_yaml_configs_tier2(config_dir, halo_repo, repo_root=repo_root)
    except Exception as exc:
        return {"name": "halo_repo", "status": "error", "detail": f"Halo Tier-2 validation failed: {exc}"}
    count = int(tier2.get("count", len(tier2.get("validated_files", []))))
    return {"name": "halo_repo", "status": "pass", "configs_validated": count, "detail": f"Halo entrypoints and Tier-2 config validation passed ({count} configs)"}


def check_arrow_dataset_dir(dataset_dir: Path, *, load_rows: bool = True) -> dict[str, Any]:
    """Verify pre-exported Halo Arrow DatasetDict structure and split readability."""
    name = f"dataset:{dataset_dir.name or dataset_dir}"
    required = [dataset_dir / "dataset_dict.json", dataset_dir / "train", dataset_dir / "test"]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        return {"name": name, "status": "fail", "detail": f"Missing Arrow artifact(s) in {dataset_dir}: {', '.join(missing)}"}
    if not load_rows:
        return {"name": name, "status": "pass", "detail": f"Arrow DatasetDict directory structure present at {dataset_dir}"}

    try:
        from datasets import load_from_disk

        ds_dict = load_from_disk(str(dataset_dir))
    except Exception as exc:
        return {"name": name, "status": "error", "detail": f"datasets.load_from_disk failed on {dataset_dir}: {exc}"}

    splits = {k: len(ds_dict[k]) for k in ds_dict.keys()}
    if splits.get("train", 0) <= 0 or splits.get("test", 0) <= 0:
        return {"name": name, "status": "fail", "splits": splits, "detail": f"Empty train/test split in {dataset_dir}: {splits}"}
    cols = list(ds_dict["train"].column_names)
    return {"name": name, "status": "pass", "splits": splits, "columns": cols, "detail": f"Loaded {dataset_dir} cleanly: {splits}"}


def check_manifest_jsonl(manifest_path: Path) -> dict[str, Any]:
    """Verify evaluation/training manifest JSONL readability and non-empty split metadata."""
    name = f"manifest:{manifest_path.name}"
    if not manifest_path.is_file():
        return {"name": name, "status": "fail", "detail": f"Manifest file not found: {manifest_path}"}
    import json

    splits: set[str] = set()
    rows = 0
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows += 1
                splits.add(str(rec.get("split") or "<missing>"))
    except Exception as exc:
        return {"name": name, "status": "error", "detail": f"Failed reading manifest {manifest_path}: {exc}"}
    if rows == 0:
        return {"name": name, "status": "fail", "rows": 0, "detail": f"Manifest {manifest_path} contains 0 rows"}
    return {"name": name, "status": "pass", "rows": rows, "splits": sorted(splits), "detail": f"{manifest_path.name}: {rows} rows (splits={sorted(splits)})"}


def run_pod_preflight(
    *,
    work_dir: Path,
    min_free_gb: float = 100.0,
    min_gpus: int = 1,
    min_vram_gb: float = 56.0,
    min_driver: str = "560.28",
    skip_gpu: bool = False,
    skip_packages: bool = False,
    halo_repo: Path | None = None,
    config_dir: Path | None = None,
    repo_root: Path | None = None,
    dataset_dirs: Sequence[Path] = (),
    manifests: Sequence[Path] = (),
    strict_versions: bool = False,
    require_hf_token: bool = False,
    load_rows: bool = True,
) -> dict[str, Any]:
    """Run all requested Dec 1 cloud pod pre-flight checks and aggregate readiness."""
    checks: list[dict[str, Any]] = []
    if not skip_gpu:
        checks.append(check_nvidia_gpus(min_gpus=min_gpus, min_vram_gb=min_vram_gb, min_driver=min_driver))
    checks.append(check_disk_space(work_dir, min_free_gb=min_free_gb))
    if not skip_packages:
        checks.append(check_python_packages(strict=strict_versions))
    checks.append(check_hf_token(required=require_hf_token))
    if halo_repo is not None and config_dir is not None:
        checks.append(check_halo_repo(halo_repo, config_dir, repo_root=repo_root))
    for ds_dir in dataset_dirs:
        checks.append(check_arrow_dataset_dir(ds_dir, load_rows=load_rows))
    for manifest in manifests:
        checks.append(check_manifest_jsonl(manifest))

    failed = [c["name"] for c in checks if c["status"] in {"fail", "error"}]
    warnings = [c["name"] for c in checks if c["status"] == "warn"]
    return {
        "ready": len(failed) == 0,
        "failed_checks": failed,
        "warning_checks": warnings,
        "checks": checks,
    }
