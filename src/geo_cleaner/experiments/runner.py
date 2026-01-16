from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .metrics import compute_metrics_from_gsm_jsonl
from .overrides import deep_merge, expand_dot_keys
from .spec import ExperimentSpec


def stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(obj: Any) -> str:
    b = stable_dumps(obj).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def load_config_dict(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text("utf-8")) or {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text("utf-8"))
    raise ValueError(f"Unsupported config format: {path}")


def write_config_json(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_dumps(cfg), encoding="utf-8")


def load_experiment_spec(path: Path) -> ExperimentSpec:
    data = yaml.safe_load(path.read_text("utf-8"))
    return ExperimentSpec.model_validate(data)


def _get_cli_disease_terms(
    defaults: Dict[str, Any], overrides: Dict[str, Any]
) -> List[str]:
    """
    Your CLI requires disease terms via --disease <term> (repeatable).
    We look for:
      - overrides["cli"]["disease"]
      - defaults["cli"]["disease"]
    """

    def _pick(x: Dict[str, Any]) -> Optional[List[str]]:
        cli = x.get("cli")
        if isinstance(cli, dict):
            d = cli.get("disease")
            if isinstance(d, list) and all(isinstance(t, str) and t.strip() for t in d):
                return [t.strip() for t in d]
        return None

    v = _pick(overrides)
    if v:
        return v
    v = _pick(defaults)
    if v:
        return v

    raise ValueError(
        "No disease terms provided. Add to experiment spec:\n"
        'defaults:\n  cli:\n    disease: ["lung cancer"]\n'
        "or per-variant overrides:\n"
        "overrides:\n  cli:\n    disease: [...]\n"
    )


def _extract_config_overrides(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports two spec styles:

    A) preferred:
       defaults:  { cli: {...}, config: {...} }
       overrides: { cli: {...}, config: {...} }

    B) legacy:
       defaults:  {...}   (treated as config overrides)
       overrides: {...}   (treated as config overrides)
    """
    cfg = d.get("config")
    if isinstance(cfg, dict):
        return cfg

    legacy = dict(d)
    legacy.pop("cli", None)
    legacy.pop("config", None)
    return legacy


def run_variant(
    *,
    cli_cmd: str,
    base_cfg: Dict[str, Any],
    exp_id: str,
    variant_id: str,
    defaults: Dict[str, Any],
    overrides: Dict[str, Any],
    runs_root: Path,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    defaults_cfg = _extract_config_overrides(defaults)
    overrides_cfg = _extract_config_overrides(overrides)

    # Apply config overrides (dot-keys supported)
    merged = deep_merge(base_cfg, expand_dot_keys(defaults_cfg))
    merged = deep_merge(merged, expand_dot_keys(overrides_cfg))

    cfg_hash = stable_hash(merged)[:12]
    run_id = f"{variant_id}_{cfg_hash}"

    exp_dir = runs_root / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_dir = exp_dir / run_id

    if run_dir.exists():
        shutil.rmtree(run_dir)

    # Stage config outside run_dir (CLI will create run_dir)
    staging_dir = exp_dir / "_staging"
    staging_dir.mkdir(exist_ok=True)
    cfg_path = staging_dir / f"{run_id}.config_effective.json"
    write_config_json(cfg_path, merged)

    disease_terms = _get_cli_disease_terms(defaults, overrides)

    # IMPORTANT: Your CLI has NO subcommand. Call geo-cleaner [OPTIONS], not geo-cleaner run [OPTIONS].
    cmd = [
        cli_cmd,
        "--config",
        str(cfg_path),
        "--outdir",
        str(exp_dir),
        "--run-id",
        run_id,
    ]
    for term in disease_terms:
        cmd += ["--disease", term]

    # Store logs outside run_dir to keep run_dir owned by CLI output layout
    logs_dir = exp_dir / "_logs" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "pipeline_stdout.log"
    stderr_path = logs_dir / "pipeline_stderr.log"

    t0 = time.time()
    with (
        stdout_path.open("w", encoding="utf-8") as out,
        stderr_path.open("w", encoding="utf-8") as err,
    ):
        proc = subprocess.run(cmd, stdout=out, stderr=err, env=env or os.environ.copy())
    t1 = time.time()

    if proc.returncode != 0:
        # Surface the tail of stderr to avoid hunting logs every time
        try:
            err_tail = stderr_path.read_text("utf-8")[-4000:]
        except Exception:
            err_tail = "<unable to read stderr log tail>"
        raise RuntimeError(
            f"Pipeline failed for {exp_id}/{variant_id} (exit={proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- stderr tail ---\n{err_tail}\n"
            f"Logs: {logs_dir}"
        )

    # Compute metrics from outputs
    gsm_jsonl = run_dir / "outputs" / "gsm.jsonl"
    if not gsm_jsonl.exists():
        raise FileNotFoundError(f"Expected GSM JSONL not found: {gsm_jsonl}")

    m = compute_metrics_from_gsm_jsonl(gsm_jsonl, top_n=50)

    metrics_payload = {
        "experiment_id": exp_id,
        "variant_id": variant_id,
        "config_hash": cfg_hash,
        "run_id": run_id,
        "wall_time_sec": float(t1 - t0),
        "n_gsm": m.n_gsm,
        "n_entities": m.n_entities,
        "yields_by_label": m.yields_by_label,
        "top_ambiguous": m.top_ambiguous,
        "top_unresolved": m.top_unresolved,
        "paths": {
            "run_dir": str(run_dir),
            "gsm_jsonl": str(gsm_jsonl),
            "manifest": str(run_dir / "manifest.json"),
            "corpus_gse_ids": str(run_dir / "corpus" / "corpus_gse_ids.json"),
            "gse_summary": str(run_dir / "reports" / "gse_summary.json"),
            "corpus_report": str(run_dir / "reports" / "corpus_report.json"),
            "logs_stdout": str(stdout_path),
            "logs_stderr": str(stderr_path),
            "config_effective_staged": str(cfg_path),
        },
        "overrides_applied": {
            "cli": {"disease": disease_terms},
            "config": expand_dot_keys(overrides_cfg),
        },
    }
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "reports" / "metrics.json").write_text(
        stable_dumps(metrics_payload), encoding="utf-8"
    )

    return run_dir


def run_experiment(
    *,
    spec_path: Path,
    base_config_path: Path,
    runs_root: Path,
    cli_cmd: str = "geo-cleaner",
    env: Optional[Dict[str, str]] = None,
) -> Path:
    spec = load_experiment_spec(spec_path)
    base_cfg = load_config_dict(base_config_path)

    exp_root = runs_root / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)

    variant_run_dirs: List[Path] = []
    for v in spec.variants:
        rd = run_variant(
            cli_cmd=cli_cmd,
            base_cfg=base_cfg,
            exp_id=spec.experiment_id,
            variant_id=v.variant_id,
            defaults=spec.defaults,
            overrides=v.overrides,
            runs_root=exp_root,
            env=env,
        )
        variant_run_dirs.append(rd)

    index = {
        "experiment_id": spec.experiment_id,
        "spec_path": str(spec_path),
        "base_config_path": str(base_config_path),
        "variant_runs": [str(p) for p in variant_run_dirs],
        "baseline_variant_id": spec.baseline_variant_id,
    }
    exp_index_path = exp_root / spec.experiment_id / "experiment_index.json"
    exp_index_path.write_text(stable_dumps(index), encoding="utf-8")

    from .consolidate import consolidate_experiment

    consolidate_experiment(exp_root / spec.experiment_id, spec)

    return exp_root / spec.experiment_id
