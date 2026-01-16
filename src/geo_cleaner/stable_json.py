from __future__ import annotations

import json
from typing import Any


def stable_dumps(obj: Any) -> str:
    # Stable dict key ordering + no whitespace (bitwise stability)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
