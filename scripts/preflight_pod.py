#!/usr/bin/env python3
"""Run Dec 1 cloud pod pre-flight readiness checks for Halo competition execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eahtr.eval_harness import write_json_atomic
from eahtr.pod_preflight import run_pod_preflight


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify GPU, disk, Python stack, Halo, datasets, and manifests before training.")
    parser.add_argument("--work-dir", type=Path, default=repo_root)
    parser.add_argument("--min-disk-gb", type=float, default=100.0)
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--min-vram-gb", type=float, default=56.0)
    parser.add_argument("--min-driver", default="560.28")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-packages", action="store_true")
    parser.add_argument("--halo-repo", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=repo_root / "configs" / "halo")
    parser.add_argument("--dataset-dir", dest="dataset_dirs", type=Path, action="append", default=[])
    parser.add_argument("--manifest", dest="manifests", type=Path, action="append", default=[])
    parser.add_argument("--strict-versions", action="store_true")
    parser.add_argument("--require-hf-token", action="store_true")
    parser.add_argument("--no-load-rows", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    report = run_pod_preflight(
        work_dir=args.work_dir,
        min_free_gb=args.min_disk_gb,
        min_gpus=args.min_gpus,
        min_vram_gb=args.min_vram_gb,
        min_driver=args.min_driver,
        skip_gpu=args.skip_gpu,
        skip_packages=args.skip_packages,
        halo_repo=args.halo_repo,
        config_dir=args.config_dir,
        repo_root=repo_root,
        dataset_dirs=args.dataset_dirs,
        manifests=args.manifests,
        strict_versions=args.strict_versions,
        require_hf_token=args.require_hf_token,
        load_rows=not args.no_load_rows,
    )
    if args.out_json is not None:
        write_json_atomic(args.out_json, report)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
