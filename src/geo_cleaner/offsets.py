from __future__ import annotations

from typing import Mapping

from .contracts import FieldOffsets, Mention


def resolve_offsets(raw_fields: Mapping[str, str], offsets: FieldOffsets) -> str:
    """
    Authoritative evidence resolver: raw_fields[field_key][start:end]
    """
    if offsets.field_key not in raw_fields:
        raise KeyError(f"Missing raw field: {offsets.field_key}")

    text = raw_fields[offsets.field_key]
    if offsets.start < 0 or offsets.end < 0 or offsets.start > offsets.end:
        raise ValueError("Invalid offsets span")

    if offsets.end > len(text):
        raise ValueError("Offsets exceed raw field length")

    return text[offsets.start : offsets.end]


def mention_roundtrip_ok(raw_fields: Mapping[str, str], mention: Mention) -> bool:
    """
    test helper: substring must equal surface_form exactly
    """
    got = resolve_offsets(raw_fields, mention.offsets())
    return got == mention.surface_form
