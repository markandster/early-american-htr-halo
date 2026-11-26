#!/usr/bin/env python3
"""Two-tier CLI validator for Halo YAML configs under configs/halo/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eahtr.halo_validate import validate_yaml_configs_tier1, validate_yaml_configs_tier2


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Halo YAML configurations.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=repo_root / "configs" / "halo",
        help="Directory containing Halo YAML configs.",
    )
    parser.add_argument(
        "--halo-repo",
        type=Path,
        default=None,
        help="Optional external whitecircle/halo checkout path for Tier-2 parser validation (Python 3.12).",
    )
    args = parser.parse_args()

    try:
        if args.halo_repo is not None:
            result = validate_yaml_configs_tier2(args.config_dir, args.halo_repo, repo_root=repo_root)
        else:
            result = validate_yaml_configs_tier1(args.config_dir, repo_root=repo_root)
    except Exception as exc:
        print(f"[ERROR] Halo config validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
