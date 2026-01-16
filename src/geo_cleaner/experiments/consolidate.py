from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .spec import ExperimentSpec
from .runner import stable_dumps


def _read_metrics(run_dir: Path) -> dict:
    p = run_dir / "reports" / "metrics.json"
    return json.loads(p.read_text("utf-8"))


def _variant_rows(exp_dir: Path) -> list[dict]:
    # exp_dir = runs/experiments/EXP-XX
    rows = []
    for rd in sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and p.name != "comparisons"]
    ):
        mp = rd / "reports" / "metrics.json"
        if mp.exists():
            rows.append(_read_metrics(rd))
    return rows


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def consolidate_experiment(exp_dir: Path, spec: ExperimentSpec) -> None:
    rows = _variant_rows(exp_dir)
    comp_dir = exp_dir / "comparisons"
    comp_dir.mkdir(exist_ok=True)

    # Always write a general summary
    summary_rows = []
    for r in rows:
        summary_rows.append(
            {
                "experiment_id": r["experiment_id"],
                "variant_id": r["variant_id"],
                "config_hash": r["config_hash"],
                "wall_time_sec": r["wall_time_sec"],
                "n_gsm": r["n_gsm"],
                "n_entities": r["n_entities"],
            }
        )
    _write_csv(
        comp_dir / "variants_summary.csv",
        [
            "experiment_id",
            "variant_id",
            "config_hash",
            "wall_time_sec",
            "n_gsm",
            "n_entities",
        ],
        summary_rows,
    )

    # Dispatch by experiment id to produce requested artifacts
    exp_id = spec.experiment_id

    if exp_id in {"EXP-02", "EXP-10"}:
        _write_linker_compare(
            comp_dir
            / (
                "compare_linker_modes.csv"
                if exp_id == "EXP-02"
                else "baseline_comparison.csv"
            ),
            rows,
        )

    elif exp_id == "EXP-03":
        _write_field_priority_ablation(comp_dir / "field_priority_ablation.csv", rows)

    elif exp_id == "EXP-04":
        _write_runtime_tradeoff(comp_dir / "runtime_tradeoff.csv", rows)

    elif exp_id == "EXP-05":
        _write_threshold_grid(comp_dir / "threshold_grid.csv", rows)

    elif exp_id == "EXP-06":
        _write_ontology_variant_ablation(
            comp_dir / "ontology_variant_ablation.csv", rows
        )

    elif exp_id == "EXP-07":
        _write_label_scope_curve(comp_dir / "label_scope_curve.csv", rows)

    elif exp_id == "EXP-08":
        _write_cross_query_summary(comp_dir / "cross_query_summary.csv", rows)

    elif exp_id == "EXP-09":
        # Cache report requested as JSON
        (comp_dir / "cache_effect_report.json").write_text(
            stable_dumps(
                {
                    "experiment_id": exp_id,
                    "variants": [
                        {
                            "variant_id": r["variant_id"],
                            "wall_time_sec": r["wall_time_sec"],
                            "n_gsm": r["n_gsm"],
                            "config_hash": r["config_hash"],
                            "paths": r.get("paths", {}),
                            "overrides_applied": r.get("overrides_applied", {}),
                        }
                        for r in rows
                    ],
                }
            ),
            encoding="utf-8",
        )


def _flatten_yields(r: dict) -> list[dict]:
    out = []
    y = r.get("yields_by_label", {}) or {}
    for label in sorted(y.keys()):
        st = y[label]
        total = sum(
            int(st.get(k, 0))
            for k in ["RESOLVED", "AMBIGUOUS", "UNRESOLVED", "REJECTED"]
        )
        out.append(
            {
                "variant_id": r["variant_id"],
                "label": label,
                "total": total,
                "resolved": int(st.get("RESOLVED", 0)),
                "ambiguous": int(st.get("AMBIGUOUS", 0)),
                "unresolved": int(st.get("UNRESOLVED", 0)),
                "rejected": int(st.get("REJECTED", 0)),
                "ambiguous_rate": (
                    (int(st.get("AMBIGUOUS", 0)) / total) if total else 0.0
                ),
                "resolved_rate": (int(st.get("RESOLVED", 0)) / total) if total else 0.0,
                "unresolved_rate": (
                    (int(st.get("UNRESOLVED", 0)) / total) if total else 0.0
                ),
            }
        )
    return out


def _write_linker_compare(path: Path, rows: list[dict]) -> None:
    flat = []
    for r in rows:
        flat.extend(_flatten_yields(r))
    header = [
        "variant_id",
        "label",
        "total",
        "resolved",
        "ambiguous",
        "unresolved",
        "rejected",
        "resolved_rate",
        "ambiguous_rate",
        "unresolved_rate",
    ]
    _write_csv(path, header, sorted(flat, key=lambda x: (x["variant_id"], x["label"])))


def _write_field_priority_ablation(path: Path, rows: list[dict]) -> None:
    # Same structure as linker compare, but we also include a short “priority_tag”
    flat = []
    for r in rows:
        tag = (r.get("overrides_applied", {}).get("textview", {}) or {}).get(
            "priority_tag", ""
        )
        for rr in _flatten_yields(r):
            rr["priority_tag"] = tag
            flat.append(rr)
    header = [
        "variant_id",
        "priority_tag",
        "label",
        "total",
        "resolved",
        "ambiguous",
        "unresolved",
        "rejected",
        "resolved_rate",
        "ambiguous_rate",
        "unresolved_rate",
    ]
    _write_csv(path, header, sorted(flat, key=lambda x: (x["variant_id"], x["label"])))


def _write_runtime_tradeoff(path: Path, rows: list[dict]) -> None:
    out = []
    for r in rows:
        o = r.get("overrides_applied", {})
        link = o.get("linker", {}) if isinstance(o.get("linker"), dict) else {}
        top_k_retrieve = link.get("top_k_retrieve", "")
        top_k_rerank = link.get("top_k_rerank", "")
        out.append(
            {
                "variant_id": r["variant_id"],
                "top_k_retrieve": top_k_retrieve,
                "top_k_rerank": top_k_rerank,
                "wall_time_sec": r["wall_time_sec"],
                "n_gsm": r["n_gsm"],
                "sec_per_gsm": (
                    (float(r["wall_time_sec"]) / r["n_gsm"]) if r["n_gsm"] else 0.0
                ),
                "n_entities": r["n_entities"],
            }
        )
    header = [
        "variant_id",
        "top_k_retrieve",
        "top_k_rerank",
        "wall_time_sec",
        "n_gsm",
        "sec_per_gsm",
        "n_entities",
    ]
    _write_csv(
        path,
        header,
        sorted(
            out, key=lambda x: (x["top_k_retrieve"], x["top_k_rerank"], x["variant_id"])
        ),
    )


def _write_threshold_grid(path: Path, rows: list[dict]) -> None:
    out = []
    for r in rows:
        pol = r.get("overrides_applied", {}).get("policy", {}) or {}
        out.append(
            {
                "variant_id": r["variant_id"],
                "tau": pol.get("tau", ""),
                "delta": pol.get("delta", ""),
                "wall_time_sec": r["wall_time_sec"],
                "n_gsm": r["n_gsm"],
                "n_entities": r["n_entities"],
            }
        )
    header = ["variant_id", "tau", "delta", "wall_time_sec", "n_gsm", "n_entities"]
    _write_csv(
        path, header, sorted(out, key=lambda x: (x["tau"], x["delta"], x["variant_id"]))
    )


def _write_ontology_variant_ablation(path: Path, rows: list[dict]) -> None:
    out = []
    for r in rows:
        ont = r.get("overrides_applied", {}).get("ontology", {}) or {}
        out.append(
            {
                "variant_id": r["variant_id"],
                "use_synonyms": ont.get("use_synonyms", ""),
                "use_definitions": ont.get("use_definitions", ""),
                "wall_time_sec": r["wall_time_sec"],
                "n_entities": r["n_entities"],
            }
        )
    header = [
        "variant_id",
        "use_synonyms",
        "use_definitions",
        "wall_time_sec",
        "n_entities",
    ]
    _write_csv(
        path,
        header,
        sorted(
            out,
            key=lambda x: (
                str(x["use_synonyms"]),
                str(x["use_definitions"]),
                x["variant_id"],
            ),
        ),
    )


def _write_label_scope_curve(path: Path, rows: list[dict]) -> None:
    out = []
    for r in rows:
        labels = (r.get("overrides_applied", {}).get("extract", {}) or {}).get(
            "labels", []
        )
        out.append(
            {
                "variant_id": r["variant_id"],
                "label_set": (
                    "|".join(labels) if isinstance(labels, list) else str(labels)
                ),
                "wall_time_sec": r["wall_time_sec"],
                "n_entities": r["n_entities"],
            }
        )
    header = ["variant_id", "label_set", "wall_time_sec", "n_entities"]
    _write_csv(path, header, sorted(out, key=lambda x: (x["variant_id"])))


def _write_cross_query_summary(path: Path, rows: list[dict]) -> None:
    out = []
    for r in rows:
        q = (r.get("overrides_applied", {}).get("query", {}) or {}).get("terms", [])
        out.append(
            {
                "variant_id": r["variant_id"],
                "query_terms": "|".join(q) if isinstance(q, list) else str(q),
                "wall_time_sec": r["wall_time_sec"],
                "n_gsm": r["n_gsm"],
                "n_entities": r["n_entities"],
            }
        )
    header = ["variant_id", "query_terms", "wall_time_sec", "n_gsm", "n_entities"]
    _write_csv(path, header, sorted(out, key=lambda x: (x["variant_id"])))
