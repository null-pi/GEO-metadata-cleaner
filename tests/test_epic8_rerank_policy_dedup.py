from geo_cleaner.contracts import (
    Candidate,
    EntityStatus,
    FieldOffsets,
    LinkedEntity,
    Mention,
)
from geo_cleaner.reranker import DummyReranker
from geo_cleaner.status_policy import assign_status
from geo_cleaner.linker import dedup_entities
from geo_cleaner.negation import is_negated, NegationConfig


def test_reranker_produces_best_score_margin_and_topN():
    rr = DummyReranker()
    cands = [
        Candidate(candidate_id="X1", candidate_label="A", score=0.6, source="vector"),
        Candidate(candidate_id="X2", candidate_label="B", score=0.9, source="vector"),
        Candidate(candidate_id="X3", candidate_label="C", score=0.7, source="vector"),
    ]
    res = rr.rerank("lung cancer", "ctx", cands)
    assert res.best is not None
    assert res.best.candidate_id == "X2"
    assert res.best_score == 0.9
    assert res.margin == 0.9 - 0.7
    assert len(res.top) == 3


def test_status_policy_thresholds_and_margin_rule():
    tau, delta = 0.7, 0.1
    assert assign_status(0.8, 0.2, tau, delta) == EntityStatus.RESOLVED
    assert assign_status(0.8, 0.05, tau, delta) == EntityStatus.AMBIGUOUS
    assert assign_status(0.6, 0.3, tau, delta) == EntityStatus.UNRESOLVED


def test_dedup_merges_by_label_and_linked_id_retaining_provenance():
    e1 = LinkedEntity(
        label="disease",
        surface_form="lung cancer",
        source_field="summary",
        offsets=FieldOffsets(field_key="summary", start=10, end=20),
        status=EntityStatus.RESOLVED,
        linked_id="DOID:1324",
        score=0.9,
        margin=0.4,
        top_candidates=[],
        provenances=[FieldOffsets(field_key="summary", start=10, end=20)],
    )
    e2 = LinkedEntity(
        label="disease",
        surface_form="lung carcinoma",
        source_field="title",
        offsets=FieldOffsets(field_key="title", start=0, end=12),
        status=EntityStatus.RESOLVED,
        linked_id="DOID:1324",
        score=0.85,
        margin=0.3,
        top_candidates=[],
        provenances=[FieldOffsets(field_key="title", start=0, end=12)],
    )

    merged = dedup_entities([e1, e2])
    assert len(merged) == 1
    assert merged[0].linked_id == "DOID:1324"
    assert len(merged[0].provenances) == 2


def test_negation_window_uses_field_scoped_offsets():
    raw = {"summary": "No lung cancer was detected."}
    start = raw["summary"].index("lung cancer")
    end = start + len("lung cancer")
    m = Mention(
        label="disease",
        surface_form="lung cancer",
        source_field="summary",
        start=start,
        end=end,
        extractor_conf=0.9,
    )
    assert is_negated(raw, m, NegationConfig(window_chars=20)) is True
