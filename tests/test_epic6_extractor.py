from geo_cleaner.extractor import RegexExtractor
from geo_cleaner.offsets import resolve_offsets


def test_extractor_outputs_mentions_with_field_offsets_and_confidence():
    raw = {"summary": "We profiled lung cancer samples and controls."}
    ex = RegexExtractor(patterns={"disease": r"\blung cancer\b"}, conf=0.9)

    ms = ex.extract(raw)
    assert len(ms) == 1
    m = ms[0]

    assert m.label == "disease"
    assert m.source_field == "summary"
    assert m.extractor_conf == 0.9
    assert m.surface_form == resolve_offsets(raw, m.offsets())


def test_mentions_reference_raw_fields_not_textview():
    raw = {
        "title": "RNA-seq",
        "summary": "lung cancer is studied",
    }
    ex = RegexExtractor(patterns={"disease": r"\blung cancer\b"})
    ms = ex.extract(raw)

    assert all(m.source_field in raw for m in ms)
    # ensure offsets are in raw field coordinates
    for m in ms:
        assert raw[m.source_field][m.start : m.end] == m.surface_form


def test_extraction_determinism_on_fixture():
    raw = {"summary": "lung cancer lung cancer"}
    ex = RegexExtractor(patterns={"disease": r"\blung cancer\b"}, conf=1.0)

    m1 = ex.extract(raw)
    m2 = ex.extract(raw)

    assert [x.model_dump() for x in m1] == [x.model_dump() for x in m2]
