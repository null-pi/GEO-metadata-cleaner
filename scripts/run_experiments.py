from __future__ import annotations

import argparse
from pathlib import Path

from geo_cleaner.experiments.runner import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "spec", type=str, help="Path to experiment YAML (e.g., experiments/EXP-02.yaml)"
    )
    ap.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Base config YAML/JSON for all variants",
    )
    ap.add_argument(
        "--runs-root", type=str, default="runs", help="Runs directory (default: runs)"
    )
    ap.add_argument(
        "--cli-cmd",
        type=str,
        default="geo-cleaner",
        help="CLI command (default: geo-cleaner)",
    )
    args = ap.parse_args()

    exp_dir = run_experiment(
        spec_path=Path(args.spec),
        base_config_path=Path(args.base_config),
        runs_root=Path(args.runs_root),
        cli_cmd=args.cli_cmd,
    )
    print(f"Wrote experiment outputs to: {exp_dir}")


if __name__ == "__main__":
    main()
