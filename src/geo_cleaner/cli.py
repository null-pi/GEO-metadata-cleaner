from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import typer

from .config import CanonicalConfig, QueryFilters, config_hash, load_config
from .manifest import RunManifest
from .ncbi_client import NCBIClient
from .pipeline import run_pipeline
from .querygse import QueryInputs, query_gse_ids
from .run_layout import create_run_dirs, make_run_layout, utc_now_iso
from .utils_device import device_info
from .utils_git import git_commit_hash

app = typer.Typer(add_completion=False)

_RUNID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_run_id(s: str) -> str:
    s = s.strip()
    s = _RUNID_SAFE.sub("_", s)
    s = s.strip("._-")
    if not s:
        raise typer.BadParameter("run-id becomes empty after sanitization.")
    return s


def _run_id(ts_utc: str, cfg_hash: str) -> str:
    safe_ts = ts_utc.replace(":", "").replace("-", "")
    return f"run_{safe_ts}_{cfg_hash[:8]}"


@app.callback(invoke_without_command=True)
def main(
    disease: List[str] = typer.Option(
        ..., "--disease", help="Disease query term(s). Repeatable."
    ),
    organism: Optional[str] = typer.Option(None, "--organism"),
    max_gse: Optional[int] = typer.Option(None, "--max-gse"),
    date_start: Optional[str] = typer.Option(
        None, "--date-start", help="YYYY or YYYY-MM or YYYY-MM-DD"
    ),
    date_end: Optional[str] = typer.Option(
        None, "--date-end", help="YYYY or YYYY-MM or YYYY-MM-DD"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", exists=True, dir_okay=False
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--outdir", help="Override output directory"
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Optional explicit run id (enables deterministic experiment runs).",
    ),
    corpus_file: Optional[Path] = typer.Option(
        None,
        "--corpus-file",
        exists=True,
        dir_okay=False,
        help="Reuse an existing corpus_gse_ids.json (offline).",
    ),
    skip_query: bool = typer.Option(
        False,
        "--skip-query",
        help="Skip NCBI QueryGSE; requires --corpus-file.",
    ),
) -> None:
    # Load config
    cfg: CanonicalConfig = load_config(config)

    # Ensure cfg.query exists
    if cfg.query is None:
        cfg.query = QueryFilters()

    # Apply CLI overrides ONLY if provided; otherwise keep config values.
    if organism is not None:
        cfg.query.organism = organism
    if date_start is not None:
        cfg.query.date_start = date_start
    if date_end is not None:
        cfg.query.date_end = date_end
    if max_gse is not None:
        cfg.query.max_gse = max_gse

    if out_dir is not None:
        cfg.run.out_dir = out_dir

    cfg_h = config_hash(cfg)
    created_at = utc_now_iso()

    chosen_run_id = _sanitize_run_id(run_id) if run_id else _run_id(created_at, cfg_h)
    layout = make_run_layout(cfg.run.out_dir, chosen_run_id)

    # Fail fast if directory exists (experiments should not mix outputs)
    if layout.run_root.exists():
        raise typer.BadParameter(f"Run directory already exists: {layout.run_root}")

    create_run_dirs(layout)

    # Write effective config for exact reproduction
    layout.config_effective_path.write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    retrieval_ts = utc_now_iso()

    # Build QueryInputs (even for skip mode, for manifest completeness)
    q_inputs = QueryInputs(
        terms=list(disease),
        organism=cfg.query.organism,
        date_start=cfg.query.date_start,
        date_end=cfg.query.date_end,
        max_gse=cfg.query.max_gse,
    )

    debug = {}
    raw_source_dir: Optional[Path] = None

    if skip_query:
        if corpus_file is None:
            raise typer.BadParameter("--skip-query requires --corpus-file")

        corpus_payload = json.loads(corpus_file.read_text("utf-8"))
        if not isinstance(corpus_payload, dict) or "gse_ids" not in corpus_payload:
            raise typer.BadParameter(
                "--corpus-file must be a corpus_gse_ids.json containing a 'gse_ids' field"
            )

        gse_ids = corpus_payload["gse_ids"]
        if not isinstance(gse_ids, list) or not all(
            isinstance(x, str) for x in gse_ids
        ):
            raise typer.BadParameter(
                "Invalid 'gse_ids' format in corpus file; expected list[str]"
            )

        retrieval_ts = corpus_payload.get("retrieved_at_utc", retrieval_ts)

        # Persist a copy into this run (self-contained artifact)
        layout.corpus_gse_ids_path.write_text(
            json.dumps(corpus_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # IMPORTANT: infer raw cache location from corpus run root:
        #   .../corpus/corpus_gse_ids.json  ->  .../<corpus_run_root>/raw
        try:
            corpus_run_root = corpus_file.parent.parent  # <corpus_run_root>/corpus/...
            candidate_raw = corpus_run_root / "raw"
            if candidate_raw.exists():
                raw_source_dir = candidate_raw
        except Exception:
            raw_source_dir = None

    else:
        # Build NCBI client only when we need network access
        ncbi = cfg.run.ncbi
        rps = ncbi.rps
        if ncbi.api_key and rps <= 3.0:
            rps = 10.0

        client = NCBIClient(
            base_url=ncbi.base_url,
            tool=ncbi.tool,
            email=ncbi.email,
            api_key=ncbi.api_key,
            timeout_s=ncbi.timeout_s,
            rps=rps,
        )

        gse_ids, debug = query_gse_ids(client, q_inputs)

        corpus_payload = {
            "retrieved_at_utc": retrieval_ts,
            "terms": list(disease),
            "filters": cfg.query.model_dump(),
            "gse_ids": gse_ids,
            "debug": debug,
        }
        layout.corpus_gse_ids_path.write_text(
            json.dumps(corpus_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # Manifest
    manifest = RunManifest(
        run_id=chosen_run_id,
        created_at_utc=created_at,
        query_terms=list(disease),
        query_filters=cfg.query.model_dump(),
        retrieval_timestamp_utc=retrieval_ts,
        code_version=git_commit_hash(),
        model_ids=cfg.models,
        ontology_versions=cfg.ontologies,
        device=device_info(),
        config_hash=cfg_h,
        config_path=str(layout.config_effective_path),
        corpus_gse_ids_path=str(layout.corpus_gse_ids_path),
        corpus_gse_count=len(gse_ids),
    )
    manifest.write(layout.manifest_path)

    # Run pipeline (real extraction/link/export)
    stats = run_pipeline(
        cfg=cfg, layout=layout, gse_ids=gse_ids, raw_source_dir=raw_source_dir
    )

    # Validate expected outputs exist (prevents “false-success” experiment runs)
    gsm_jsonl = layout.run_root / "outputs" / "gsm.jsonl"
    gse_summary = layout.run_root / "reports" / "gse_summary.json"
    corpus_report = layout.run_root / "reports" / "corpus_report.json"

    missing = [p for p in (gsm_jsonl, gse_summary, corpus_report) if not p.exists()]
    if missing:
        raise typer.BadParameter(f"Pipeline finished but missing outputs: {missing}")

    typer.echo(
        f"Processed GSE={stats.gse_processed} GSM={stats.gsm_processed} entities={stats.entities_total}"
    )
    typer.echo(f"Run created: {layout.run_root}")
    typer.echo(f"Manifest:   {layout.manifest_path}")
    typer.echo(f"Corpus:     {layout.corpus_gse_ids_path} ({len(gse_ids)} GSEs)")
