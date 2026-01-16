from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import Mention
from .offsets import resolve_offsets


@dataclass(frozen=True)
class NegationConfig:
    window_chars: int = 60


_NEG_PATTERNS = [
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bwithout\b", re.IGNORECASE),
    re.compile(r"\bnegative for\b", re.IGNORECASE),
    re.compile(r"\bno evidence of\b", re.IGNORECASE),
]


def is_negated(
    raw_fields: dict[str, str], mention: Mention, cfg: NegationConfig = NegationConfig()
) -> bool:
    text = raw_fields.get(mention.source_field, "")
    if not text:
        return False
    start = max(0, mention.start - cfg.window_chars)
    end = min(len(text), mention.end + cfg.window_chars)
    window = text[start:end]
    return any(p.search(window) for p in _NEG_PATTERNS)
