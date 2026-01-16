from pathlib import Path

import pytest
from jsonschema import validate as js_validate

from geo_cleaner.contracts import Candidate, EntityStatus, FieldOffsets, LinkedEntity
from geo_cleaner.export_models import CorpusReport, GSMCleanedRecord, GSESummary
from geo_cleaner.exporter import (
    compute_corpus_report,
    compute_gse_summary,
    group_entities_by_label,
    write_gsm_jsonl,
)


def _mk_entity(
    label: str,
    field: str,
    start: int,
    end: int,
    sf: str,
    status: EntityStatus,
    linked_id=None,
):
    return LinkedEntity(
        label=label,
        surface_form=sf,
        source_field=field,
        offsets=FieldOffsets(field_key=field, start=start, end=end),
        status=status,
        linked_id=linked_id,
        score=0.9 if status == EntityStatus.RESOLVED else 1.0,
        margin=0.4 if status == EntityStatus.RESOLVED else 0.0,
        top_candidates=(
            [
                Candidate(
                    candidate_id=linked_id or "X:0",
                    candidate_label=sf,
                    score=0.9,
                    source="lexical_exact",
                )
            ]
            if linked_id
            else []
        ),
        provenances=[FieldOffsets(field_key=field, start=start, end=end)],
    )


def test_gsm_jsonl_record_has_required_fields(tmp_path: Path):
    ents = [
        _mk_entity(
            "disease",
            "summary",
            12,
            23,
            "lung cancer",
            EntityStatus.RESOLVED,
            linked_id="DOID:1324",
        ),
        _mk_entity(
            "disease", "title", 0, 6, "cancer", EntityStatus.AMBIGUOUS, linked_id=None
        ),
    ]
    by_label = group_entities_by_label(ents)

    rec = GSMCleanedRecord(
        gse_id="GSE1",
        gsm_id="GSM1",
        textview_hash="abc123",
        textview_fields=None,
        entities=by_label,
    )

    out = tmp_path / "gsm.jsonl"
    write_gsm_jsonl([rec], out)

    line = out.read_text("utf-8").splitlines()[0]
    assert '"gse_id":"GSE1"' in line
    assert '"gsm_id":"GSM1"' in line
    assert '"textview_hash":"abc123"' in line
    assert '"entities"' in line


def test_gse_summary_contains_yields_and_top_ambiguous_unresolved():
    r1 = GSMCleanedRecord(
        gse_id="GSE1",
        gsm_id="GSM1",
        textview_hash="h1",
        entities=group_entities_by_label(
            [
                _mk_entity(
                    "disease",
                    "summary",
                    12,
                    23,
                    "lung cancer",
                    EntityStatus.RESOLVED,
                    "DOID:1324",
                ),
                _mk_entity(
                    "disease", "title", 0, 6, "cancer", EntityStatus.AMBIGUOUS, None
                ),
            ]
        ),
    )
    r2 = GSMCleanedRecord(
        gse_id="GSE1",
        gsm_id="GSM2",
        textview_hash="h2",
        entities=group_entities_by_label(
            [
                _mk_entity(
                    "disease",
                    "summary",
                    0,
                    13,
                    "unknown disease",
                    EntityStatus.UNRESOLVED,
                    None,
                ),
            ]
        ),
    )

    s = compute_gse_summary("GSE1", [r1, r2], top_n=10)
    assert s.n_gsm_processed == 2
    assert "disease" in s.yields_by_label
    assert s.yields_by_label["disease"].resolved == 1
    assert s.yields_by_label["disease"].ambiguous == 1
    assert s.yields_by_label["disease"].unresolved == 1

    assert any(x.surface_form == "cancer" for x in s.top_ambiguous)
    assert any(x.surface_form == "unknown disease" for x in s.top_unresolved)


def test_corpus_report_contains_selection_and_aggregate_stats():
    # One GSE processed
    gse_summary = GSESummary(
        gse_id="GSE1",
        n_gsm_processed=2,
        yields_by_label={
            "disease": {
                "total": 3,
                "resolved": 1,
                "ambiguous": 1,
                "unresolved": 1,
                "rejected": 0,
            }
        },
        top_ambiguous=[],
        top_unresolved=[],
    )

    rep = compute_corpus_report(
        run_id="run_TEST",
        query_terms=["lung cancer"],
        query_filters={"organism": "Homo sapiens", "max_gse": 200},
        manifest_path="runs/run_TEST/manifest.json",
        corpus_gse_ids_path="runs/run_TEST/corpus/corpus_gse_ids.json",
        gse_selected=["GSE1", "GSE2"],
        gse_to_summary={"GSE1": gse_summary},
        resources={"models": {"reranker": "dummy"}, "ontologies": {"doid": "hash"}},
    )

    assert rep.n_gse_selected == 2
    assert rep.n_gse_processed == 1
    assert rep.gse_processed == ["GSE1"]
    assert "disease" in rep.aggregate_yields_by_label


def test_schema_validation_on_all_outputs():
    gsm_schema = GSMCleanedRecord.model_json_schema()
    gse_schema = GSESummary.model_json_schema()
    corpus_schema = CorpusReport.model_json_schema()

    gsm_payload = {
        "schema_version": "1.0",
        "gse_id": "GSE1",
        "gsm_id": "GSM1",
        "textview_hash": "h",
        "textview_fields": None,
        "entities": {},
    }
    js_validate(gsm_payload, gsm_schema)

    gse_payload = {
        "schema_version": "1.0",
        "gse_id": "GSE1",
        "n_gsm_processed": 0,
        "yields_by_label": {},
        "top_ambiguous": [],
        "top_unresolved": [],
    }
    js_validate(gse_payload, gse_schema)

    corpus_payload = {
        "schema_version": "1.0",
        "run_id": "run_TEST",
        "query_terms": [],
        "query_filters": {},
        "manifest_path": "m.json",
        "corpus_gse_ids_path": "c.json",
        "n_gse_selected": 0,
        "n_gse_processed": 0,
        "gse_selected": [],
        "gse_processed": [],
        "aggregate_yields_by_label": {},
        "global_top_ambiguous": [],
        "global_top_unresolved": [],
        "resources": {},
    }
    js_validate(corpus_payload, corpus_schema)
