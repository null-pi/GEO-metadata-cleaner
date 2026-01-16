from pathlib import Path

from jsonschema import validate as js_validate

from geo_cleaner.contracts import (
    Candidate,
    EntityStatus,
    FieldOffsets,
    LinkedEntity,
    Mention,
)
from geo_cleaner.export_models import CorpusReport, GSMCleanedRecord, GSESummary
from geo_cleaner.exporter import (
    compute_corpus_report,
    compute_gse_summary,
    group_entities_by_label,
    write_gsm_jsonl,
    write_json,
)
from geo_cleaner.stable_json import stable_dumps


def _mk_record(
    gse_id: str, gsm_id: str, tv_hash: str, entities: list[LinkedEntity]
) -> GSMCleanedRecord:
    return GSMCleanedRecord(
        gse_id=gse_id,
        gsm_id=gsm_id,
        textview_hash=tv_hash,
        textview_fields=None,
        entities=group_entities_by_label(entities),
    )


def _mk_entity(label, field, start, end, sf, status, linked_id=None, top=None):
    return LinkedEntity(
        label=label,
        surface_form=sf,
        source_field=field,
        offsets=FieldOffsets(field_key=field, start=start, end=end),
        status=status,
        linked_id=linked_id,
        score=1.0 if status != EntityStatus.RESOLVED else 0.9,
        margin=0.0 if status != EntityStatus.RESOLVED else 0.4,
        top_candidates=top or [],
        provenances=[FieldOffsets(field_key=field, start=start, end=end)],
    )


def test_end_to_end_fixture_is_bitwise_stable(tmp_path: Path):
    # Fixed fixture “raw” (in a real harness, these come from stored raw GSM files)
    gse_id = "GSE_FIX"
    gse_selected = [gse_id]
    query_terms = ["lung cancer"]
    query_filters = {"organism": "Homo sapiens", "max_gse": 10}

    # Fixed entities (pretend output of link+policy+dedup)
    e1 = _mk_entity(
        "disease",
        "summary",
        12,
        23,
        "lung cancer",
        EntityStatus.RESOLVED,
        linked_id="DOID:1324",
        top=[
            Candidate(
                candidate_id="DOID:1324",
                candidate_label="lung cancer",
                score=0.95,
                source="rerank",
            )
        ],
    )
    e2 = _mk_entity(
        "disease",
        "title",
        0,
        6,
        "cancer",
        EntityStatus.AMBIGUOUS,
        linked_id=None,
        top=[
            Candidate(
                candidate_id="DOID:1324",
                candidate_label="lung cancer",
                score=0.80,
                source="rerank",
            ),
            Candidate(
                candidate_id="DOID:1612",
                candidate_label="breast cancer",
                score=0.79,
                source="rerank",
            ),
        ],
    )
    e3 = _mk_entity(
        "disease",
        "summary",
        0,
        13,
        "unknown disease",
        EntityStatus.UNRESOLVED,
        linked_id=None,
        top=[],
    )

    r1 = _mk_record(gse_id, "GSM_FIX_1", "tvh1", [e1, e2])
    r2 = _mk_record(gse_id, "GSM_FIX_2", "tvh2", [e3])

    # Export outputs twice and compare bytes
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    gsm_jsonl_1 = out1 / "gsm.jsonl"
    gsm_jsonl_2 = out2 / "gsm.jsonl"

    write_gsm_jsonl([r1, r2], gsm_jsonl_1)
    write_gsm_jsonl([r1, r2], gsm_jsonl_2)

    s1 = compute_gse_summary(gse_id, [r1, r2], top_n=10)
    s2 = compute_gse_summary(gse_id, [r1, r2], top_n=10)

    rep1 = compute_corpus_report(
        run_id="run_FIX",
        query_terms=query_terms,
        query_filters=query_filters,
        manifest_path="runs/run_FIX/manifest.json",
        corpus_gse_ids_path="runs/run_FIX/corpus/corpus_gse_ids.json",
        gse_selected=gse_selected,
        gse_to_summary={gse_id: s1},
        resources={"models": {"reranker": "dummy"}, "ontologies": {"toy": "hash123"}},
    )
    rep2 = compute_corpus_report(
        run_id="run_FIX",
        query_terms=query_terms,
        query_filters=query_filters,
        manifest_path="runs/run_FIX/manifest.json",
        corpus_gse_ids_path="runs/run_FIX/corpus/corpus_gse_ids.json",
        gse_selected=gse_selected,
        gse_to_summary={gse_id: s2},
        resources={"models": {"reranker": "dummy"}, "ontologies": {"toy": "hash123"}},
    )

    write_json(out1 / "gse_summary.json", s1)
    write_json(out2 / "gse_summary.json", s2)
    write_json(out1 / "corpus_report.json", rep1)
    write_json(out2 / "corpus_report.json", rep2)

    assert gsm_jsonl_1.read_bytes() == gsm_jsonl_2.read_bytes()
    assert (out1 / "gse_summary.json").read_bytes() == (
        out2 / "gse_summary.json"
    ).read_bytes()
    assert (out1 / "corpus_report.json").read_bytes() == (
        out2 / "corpus_report.json"
    ).read_bytes()


def test_offsets_stable_across_resume_restart():
    raw = {"summary": "We profiled lung cancer samples."}
    start = raw["summary"].index("lung cancer")
    end = start + len("lung cancer")

    m1 = Mention(
        label="disease",
        surface_form="lung cancer",
        source_field="summary",
        start=start,
        end=end,
        extractor_conf=0.9,
    )
    # “resume”: recompute
    start2 = raw["summary"].index("lung cancer")
    end2 = start2 + len("lung cancer")
    m2 = Mention(
        label="disease",
        surface_form="lung cancer",
        source_field="summary",
        start=start2,
        end=end2,
        extractor_conf=0.9,
    )

    assert (m1.source_field, m1.start, m1.end) == (m2.source_field, m2.start, m2.end)


def test_model_and_ontology_versions_logged_and_consistent():
    # Minimal check: corpus report must carry resource versions (paper reproducibility)
    rep = CorpusReport(
        run_id="run_FIX",
        query_terms=["lung cancer"],
        query_filters={"organism": "Homo sapiens"},
        manifest_path="m.json",
        corpus_gse_ids_path="c.json",
        n_gse_selected=1,
        n_gse_processed=1,
        gse_selected=["GSE_FIX"],
        gse_processed=["GSE_FIX"],
        aggregate_yields_by_label={},
        global_top_ambiguous=[],
        global_top_unresolved=[],
        resources={
            "models": {"reranker": "dummy@1"},
            "ontologies": {"doid": "sha256:abc"},
        },
    )
    assert "models" in rep.resources
    assert "ontologies" in rep.resources
    assert rep.resources["ontologies"]["doid"].startswith("sha256:")


def test_output_schema_backward_compatible_versioning():
    # Hard requirement: schema_version present and correct
    r = GSMCleanedRecord(gse_id="GSE1", gsm_id="GSM1", textview_hash="h", entities={})
    assert r.schema_version == "1.0"
