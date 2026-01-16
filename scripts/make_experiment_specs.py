from __future__ import annotations

from pathlib import Path
import yaml


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def main():
    out = Path("experiments")
    out.mkdir(exist_ok=True)

    # EXP-01 baseline
    write(
        out / "EXP-01.yaml",
        {
            "experiment_id": "EXP-01",
            "description": "Baseline end-to-end (single config)",
            "defaults": {},
            "variants": [{"variant_id": "baseline", "overrides": {}}],
            "export": {},
        },
    )

    # EXP-02 linker ablation
    write(
        out / "EXP-02.yaml",
        {
            "experiment_id": "EXP-02",
            "description": "Lexical vs bi-encoder vs bi-encoder+cross rerank",
            "variants": [
                {
                    "variant_id": "lexical_only",
                    "overrides": {"linker.mode": "lexical_only"},
                },
                {
                    "variant_id": "lexical_plus_bi",
                    "overrides": {
                        "linker.mode": "lexical_plus_bi",
                        "models.embedder": "sentence-transformers/all-MiniLM-L6-v2",
                    },
                },
                {
                    "variant_id": "lexical_plus_bi_plus_cross",
                    "overrides": {
                        "linker.mode": "lexical_plus_bi_plus_cross",
                        "models.embedder": "sentence-transformers/all-MiniLM-L6-v2",
                        "models.reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    },
                },
            ],
            "export": {"csv": "compare_linker_modes.csv"},
            "baseline_variant_id": "lexical_only",
        },
    )

    # EXP-03 TextView field priority sensitivity
    write(
        out / "EXP-03.yaml",
        {
            "experiment_id": "EXP-03",
            "description": "TextView field priority sensitivity (meaningful if reranker uses TextView context)",
            "variants": [
                {
                    "variant_id": "prio_char_source_title_desc",
                    "overrides": {
                        "textview.priority_tag": "char>source>title>desc",
                        "textview.field_priority": [
                            "characteristics_ch1",
                            "source_name_ch1",
                            "title",
                            "summary",
                        ],
                        "linker.context_mode": "textview",
                    },
                },
                {
                    "variant_id": "prio_title_char_desc",
                    "overrides": {
                        "textview.priority_tag": "title>char>desc",
                        "textview.field_priority": [
                            "title",
                            "characteristics_ch1",
                            "summary",
                        ],
                        "linker.context_mode": "textview",
                    },
                },
            ],
            "export": {"csv": "field_priority_ablation.csv"},
        },
    )

    # EXP-04 Candidate set size and rerank depth
    variants = []
    for k in [10, 25, 50, 100]:
        for r in [5, 10, 20]:
            variants.append(
                {
                    "variant_id": f"k{k}_r{r}",
                    "overrides": {"linker.top_k_retrieve": k, "linker.top_k_rerank": r},
                }
            )
    write(
        out / "EXP-04.yaml",
        {
            "experiment_id": "EXP-04",
            "description": "top_k_retrieve x top_k_rerank tradeoff",
            "variants": variants,
            "export": {"csv": "runtime_tradeoff.csv", "plots": True},
        },
    )

    # EXP-05 threshold/margin sweep
    variants = []
    for tau in [0.6, 0.7, 0.8]:
        for delta in [0.05, 0.1, 0.2]:
            variants.append(
                {
                    "variant_id": f"tau{tau}_d{delta}",
                    "overrides": {"policy.tau": tau, "policy.delta": delta},
                }
            )
    write(
        out / "EXP-05.yaml",
        {
            "experiment_id": "EXP-05",
            "description": "tau/delta grid for deterministic status policy calibration",
            "variants": variants,
            "export": {"csv": "threshold_grid.csv"},
        },
    )

    # EXP-06 ontology variants
    write(
        out / "EXP-06.yaml",
        {
            "experiment_id": "EXP-06",
            "description": "Synonyms/definitions on/off ablation",
            "variants": [
                {
                    "variant_id": "syn_on_def_on",
                    "overrides": {
                        "ontology.use_synonyms": True,
                        "ontology.use_definitions": True,
                    },
                },
                {
                    "variant_id": "syn_on_def_off",
                    "overrides": {
                        "ontology.use_synonyms": True,
                        "ontology.use_definitions": False,
                    },
                },
                {
                    "variant_id": "syn_off_def_on",
                    "overrides": {
                        "ontology.use_synonyms": False,
                        "ontology.use_definitions": True,
                    },
                },
                {
                    "variant_id": "syn_off_def_off",
                    "overrides": {
                        "ontology.use_synonyms": False,
                        "ontology.use_definitions": False,
                    },
                },
            ],
            "export": {"csv": "ontology_variant_ablation.csv"},
        },
    )

    # EXP-07 label scope expansion
    write(
        out / "EXP-07.yaml",
        {
            "experiment_id": "EXP-07",
            "description": "Incremental label scope curve",
            "variants": [
                {
                    "variant_id": "set_A",
                    "overrides": {"extract.labels": ["disease", "tissue", "organism"]},
                },
                {
                    "variant_id": "set_B",
                    "overrides": {
                        "extract.labels": ["disease", "tissue", "organism", "cell_type"]
                    },
                },
                {
                    "variant_id": "set_C",
                    "overrides": {
                        "extract.labels": [
                            "disease",
                            "tissue",
                            "organism",
                            "cell_type",
                            "drug",
                        ]
                    },
                },
                {
                    "variant_id": "set_D",
                    "overrides": {
                        "extract.labels": [
                            "disease",
                            "tissue",
                            "organism",
                            "cell_type",
                            "drug",
                            "cell_line",
                            "assay",
                        ]
                    },
                },
            ],
            "export": {"csv": "label_scope_curve.csv"},
        },
    )

    # EXP-08 robustness across disease queries
    write(
        out / "EXP-08.yaml",
        {
            "experiment_id": "EXP-08",
            "description": "Panel of disease queries (robustness)",
            "variants": [
                {
                    "variant_id": "lung_cancer",
                    "overrides": {"query.terms": ["lung cancer"]},
                },
                {
                    "variant_id": "rheumatoid_arthritis",
                    "overrides": {"query.terms": ["rheumatoid arthritis"]},
                },
                {
                    "variant_id": "influenza",
                    "overrides": {"query.terms": ["influenza"]},
                },
                {
                    "variant_id": "alzheimers",
                    "overrides": {"query.terms": ["Alzheimer disease"]},
                },
            ],
            "export": {"csv": "cross_query_summary.csv"},
        },
    )

    # EXP-09 resume/caching effectiveness
    # Assumes your pipeline uses a configurable cache dir; if not, add `cache.dir` to config and honor it in OntologyBundle.
    write(
        out / "EXP-09.yaml",
        {
            "experiment_id": "EXP-09",
            "description": "Cold vs warm cache + partial restart (requires run-dir resume semantics)",
            "variants": [
                {"variant_id": "cold_run", "overrides": {"cache.mode": "cold"}},
                {"variant_id": "warm_run", "overrides": {"cache.mode": "warm"}},
                {"variant_id": "partial_restart", "overrides": {"run.resume": True}},
            ],
            "export": {"json": "cache_effect_report.json"},
        },
    )

    # EXP-10 baseline comparisons (same modes as EXP-02 but framed as baselines)
    write(
        out / "EXP-10.yaml",
        {
            "experiment_id": "EXP-10",
            "description": "Baselines: dictionary-only vs bi-encoder-only vs full rerank",
            "variants": [
                {
                    "variant_id": "dictionary_only",
                    "overrides": {"linker.mode": "lexical_only"},
                },
                {
                    "variant_id": "bi_encoder_only",
                    "overrides": {
                        "linker.mode": "lexical_plus_bi",
                        "models.embedder": "sentence-transformers/all-MiniLM-L6-v2",
                    },
                },
                {
                    "variant_id": "full_rerank",
                    "overrides": {
                        "linker.mode": "lexical_plus_bi_plus_cross",
                        "models.embedder": "sentence-transformers/all-MiniLM-L6-v2",
                        "models.reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    },
                },
            ],
            "export": {"csv": "baseline_comparison.csv"},
        },
    )

    print("Wrote experiments/EXP-01.yaml ... experiments/EXP-10.yaml")


if __name__ == "__main__":
    main()
