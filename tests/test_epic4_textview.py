from geo_cleaner.textview import build_textview


def test_textview_deterministic_given_same_raw_input():
    raw1 = {"summary": "B", "title": "A"}  # intentionally unordered insertion
    raw2 = {"title": "A", "summary": "B"}

    prio = ["title", "summary"]
    tv1 = build_textview(raw1, prio)
    tv2 = build_textview(raw2, prio)

    assert tv1.concatenated_text == tv2.concatenated_text
    assert tv1.hash == tv2.hash


def test_textview_field_priority_applied():
    raw = {"summary": "B", "title": "A"}
    prio = ["summary", "title"]
    tv = build_textview(raw, prio)

    keys = [sf.field_key for sf in tv.fields_selected]
    assert keys == ["summary", "title"]


def test_textview_hash_changes_if_selected_fields_change():
    raw = {"title": "A", "summary": "B"}
    prio = ["title", "summary"]

    tv1 = build_textview(raw, prio)
    raw2 = {"title": "A", "summary": "B changed"}
    tv2 = build_textview(raw2, prio)

    assert tv1.hash != tv2.hash
