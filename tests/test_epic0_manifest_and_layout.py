import json
from pathlib import Path

from geo_cleaner.config import CanonicalConfig, config_hash
from geo_cleaner.manifest import RunManifest
from geo_cleaner.run_layout import create_run_dirs, make_run_layout


def test_config_hash_stable(tmp_path: Path):
    cfg1 = CanonicalConfig()
    cfg2 = CanonicalConfig()
    assert config_hash(cfg1) == config_hash(cfg2)


def test_cli_creates_run_directory_layout(tmp_path: Path):
    layout = make_run_layout(tmp_path, "run_TEST_00000000")
    create_run_dirs(layout)

    assert layout.run_root.exists()
    assert layout.corpus_dir.exists()
    assert layout.logs_dir.exists()
    assert layout.cache_dir.exists()
    assert layout.raw_dir.exists()
    assert layout.outputs_dir.exists()
    assert layout.reports_dir.exists()


def test_run_manifest_written_with_required_fields(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    m = RunManifest(
        run_id="run_TEST",
        created_at_utc="2026-01-16T00:00:00+00:00",
        query_terms=["lung cancer"],
        query_filters={
            "organism": "Homo sapiens",
            "date_start": None,
            "date_end": None,
            "max_gse": 200,
        },
        retrieval_timestamp_utc="2026-01-16T00:00:01+00:00",
        code_version="deadbeef",
        model_ids={"mention_extractor": "stub"},
        ontology_versions={"doid": "2025-01"},
        device={"cpu": "x"},
        config_hash="abc",
        config_path="config_effective.json",
        corpus_gse_ids_path="corpus/corpus_gse_ids.json",
        corpus_gse_count=12,
    )
    m.write(manifest_path)

    payload = json.loads(manifest_path.read_text("utf-8"))
    for k in [
        "run_id",
        "created_at_utc",
        "query_terms",
        "query_filters",
        "retrieval_timestamp_utc",
        "code_version",
        "model_ids",
        "ontology_versions",
        "device",
        "config_hash",
        "config_path",
    ]:
        assert k in payload
