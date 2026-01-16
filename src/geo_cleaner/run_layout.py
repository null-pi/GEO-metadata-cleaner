from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    run_root: Path
    manifest_path: Path
    config_effective_path: Path
    corpus_dir: Path
    corpus_gse_ids_path: Path
    logs_dir: Path
    cache_dir: Path
    raw_dir: Path
    outputs_dir: Path
    reports_dir: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_layout(out_dir: Path, run_id: str) -> RunLayout:
    run_root = out_dir / run_id
    corpus_dir = run_root / "corpus"
    logs_dir = run_root / "logs"
    cache_dir = run_root / "cache"
    raw_dir = run_root / "raw"
    outputs_dir = run_root / "outputs"
    reports_dir = run_root / "reports"

    return RunLayout(
        run_root=run_root,
        manifest_path=run_root / "manifest.json",
        config_effective_path=run_root / "config_effective.json",
        corpus_dir=corpus_dir,
        corpus_gse_ids_path=corpus_dir / "corpus_gse_ids.json",
        logs_dir=logs_dir,
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        outputs_dir=outputs_dir,
        reports_dir=reports_dir,
    )


def create_run_dirs(layout: RunLayout) -> None:
    layout.run_root.mkdir(parents=True, exist_ok=False)
    for d in [
        layout.corpus_dir,
        layout.logs_dir,
        layout.cache_dir,
        layout.raw_dir,
        layout.outputs_dir,
        layout.reports_dir,
    ]:
        d.mkdir(parents=True, exist_ok=False)
