import json

import pytest
from jsonschema import validate as js_validate

from geo_cleaner.contracts import Candidate, EntityStatus, FieldOffsets, LinkedEntity, Mention
from geo_cleaner.offsets import mention_roundtrip_ok, resolve_offsets


def test_mention_offsets_roundtrip_into_raw_field_text():
    raw = {
        "title": "RNA-seq of lung cancer tissue",
        "summary": "We profiled lung cancer samples and matched controls.",
    }
    s = raw["summary"]
    start = s.index("lung cancer")
    end = start + len("lung cancer")

    m = Mention(
        label="disease",
        surface_form="lung cancer",
        source_field="summary",
        start=start,
        end=end,
        extractor_conf=0.91,
    )
    assert mention_roundtrip_ok(raw, m)
    assert resolve_offsets(raw, m.offsets()) == "lung cancer"


def test_linked_entity_minimum_fields_present():
    e = LinkedEntity(
        label="disease",
        surface_form="lung cancer",
        source_field="summary",
        offsets=FieldOffsets(field_key="summary", start=13, end=23),
        status=EntityStatus.AMBIGUOUS,
        linked_id=None,
        score=0.62,
        margin=0.03,
        top_candidates=[
            Candidate(candidate_id="DOID:1324", candidate_label="lung cancer", score=0.62),
            Candidate(candidate_id="DOID:3908", candidate_label="lung carcinoma", score=0.59),
        ],
    )
    d = e.model_dump()
    for k in ["label", "surface_form", "source_field", "offsets", "status", "top_candidates"]:
        assert k in d


def test_json_schema_validation_for_entity_payload():
    schema = LinkedEntity.model_json_schema()

    unresolved_payload = {
        "label": "disease",
        "surface_form": "unknown syndrome",
        "source_field": "summary",
        "offsets": {"field_key": "summary", "start": 0, "end": 15},
        "status": "UNRESOLVED",
        "linked_id": None,
        "score": None,
        "margin": None,
        "top_candidates": [],
    }
    js_validate(instance=unresolved_payload, schema=schema)

    resolved_payload = {
        "label": "disease",
        "surface_form": "lung cancer",
        "source_field": "summary",
        "offsets": {"field_key": "summary", "start": 13, "end": 23},
        "status": "RESOLVED",
        "linked_id": "DOID:1324",
        "score": 0.91,
        "margin": 0.40,
        "top_candidates": [
            {"candidate_id": "DOID:1324", "candidate_label": "lung cancer", "score": 0.91}
        ],
    }
    js_validate(instance=resolved_payload, schema=schema)

    # This must fail: RESOLVED without linked_id
    bad_payload = dict(resolved_payload)
    bad_payload["linked_id"] = None
    with pytest.raises(Exception):
        LinkedEntity.model_validate(bad_payload)
