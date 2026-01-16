from __future__ import annotations

import copy
from typing import Any, Dict


def _set_path(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def expand_dot_keys(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allows overrides like {"policy.tau": 0.75, "textview.field_priority": [...]}.
    If overrides are already nested dicts, they pass through.
    """
    out: Dict[str, Any] = {}
    for k, v in overrides.items():
        if "." in k:
            _set_path(out, k, v)
        else:
            out[k] = v
    return out


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge dicts. Lists are replaced (not merged).
    """
    b = copy.deepcopy(base)
    o = copy.deepcopy(override)

    def rec(x: Any, y: Any) -> Any:
        if isinstance(x, dict) and isinstance(y, dict):
            for k, v in y.items():
                x[k] = rec(x.get(k), v)
            return x
        # lists/atomics replace
        return y

    return rec(b, o)
