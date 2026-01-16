from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

from pydantic import BaseModel, Field


class SelectedField(BaseModel):
    field_key: str
    text: str


class TextView(BaseModel):
    fields_selected: list[SelectedField]
    concatenated_text: str
    hash: str


def _stable_hash(obj) -> str:
    b = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def build_textview(
    gsm_raw: Mapping[str, str],
    field_priority: Sequence[str],
    max_field_chars: int = 4000,
) -> TextView:
    selected: list[SelectedField] = []

    for key in field_priority:
        if key not in gsm_raw:
            continue
        v = gsm_raw[key]
        if v is None:
            continue
        v = str(v)
        if not v.strip():
            continue
        if max_field_chars and len(v) > max_field_chars:
            v = v[:max_field_chars]
        selected.append(SelectedField(field_key=key, text=v))

    # Deterministic formatting for context
    parts = []
    for sf in selected:
        parts.append(f"[{sf.field_key}]\n{sf.text}")
    concat = "\n\n".join(parts)

    # Hash is a function of the selected fields and their order/content
    h = _stable_hash([sf.model_dump() for sf in selected])

    return TextView(fields_selected=selected, concatenated_text=concat, hash=h)
