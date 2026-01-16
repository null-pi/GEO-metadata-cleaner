from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .contracts import EntityStatus


@dataclass(frozen=True)
class PolicyConfig:
    tau: float = 0.70  # best-score threshold
    delta: float = 0.10  # margin threshold
    top_n: int = 5  # candidates retained when ambiguous/unresolved


def compute_margin(sorted_scores: List[float]) -> float:
    if not sorted_scores:
        return 0.0
    if len(sorted_scores) == 1:
        return 1.0  # deterministic "high margin" when only one candidate exists
    return float(sorted_scores[0] - sorted_scores[1])


def assign_status(best: float, margin: float, tau: float, delta: float) -> EntityStatus:
    if best >= tau and margin >= delta:
        return EntityStatus.RESOLVED
    if best >= tau and margin < delta:
        return EntityStatus.AMBIGUOUS
    return EntityStatus.UNRESOLVED
