from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class Metrics:
    n_gsm: int
    n_entities: int
    yields_by_label: Dict[str, Dict[str, int]]  # label -> status -> count
    top_ambiguous: list[dict]
    top_unresolved: list[dict]


def _inc(d: Dict[str, Dict[str, int]], label: str, status: str) -> None:
    if label not in d:
        d[label] = {"RESOLVED": 0, "AMBIGUOUS": 0, "UNRESOLVED": 0, "REJECTED": 0}
    if status not in d[label]:
        d[label][status] = 0
    d[label][status] += 1


def compute_metrics_from_gsm_jsonl(
    gsm_jsonl_path: Path,
    top_n: int = 50,
) -> Metrics:
    n_gsm = 0
    n_entities = 0

    yields_by_label: Dict[str, Dict[str, int]] = {}
    amb_counter: Counter[Tuple[str, str, str]] = Counter()
    unr_counter: Counter[Tuple[str, str, str]] = Counter()

    with gsm_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_gsm += 1
            obj = json.loads(line)

            entities = obj.get("entities", {}) or {}
            for label, ents in entities.items():
                for e in ents or []:
                    status = e.get("status", "UNRESOLVED")
                    _inc(yields_by_label, label, status)
                    n_entities += 1

                    sf = e.get("surface_form", "")
                    field = e.get("source_field", "")
                    if status == "AMBIGUOUS":
                        amb_counter[(label, field, sf)] += 1
                    elif status == "UNRESOLVED":
                        unr_counter[(label, field, sf)] += 1

    def top(counter: Counter[Tuple[str, str, str]]) -> list[dict]:
        items = sorted(
            counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2])
        )
        out = []
        for (label, field, sf), cnt in items[:top_n]:
            out.append(
                {
                    "label": label,
                    "source_field": field,
                    "surface_form": sf,
                    "count": int(cnt),
                }
            )
        return out

    return Metrics(
        n_gsm=n_gsm,
        n_entities=n_entities,
        yields_by_label={k: yields_by_label[k] for k in sorted(yields_by_label.keys())},
        top_ambiguous=top(amb_counter),
        top_unresolved=top(unr_counter),
    )


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text("utf-8"))
