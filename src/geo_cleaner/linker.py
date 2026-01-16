from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .candidate_retrieval import CandidateRetriever
from .contracts import Candidate, EntityStatus, FieldOffsets, LinkedEntity, Mention
from .status_policy import PolicyConfig, assign_status
from .reranker import BaseReranker
from .ontology_bundle import normalize_text
from .negation import is_negated, NegationConfig


def local_context(
    raw_fields: Dict[str, str], m: Mention, window_chars: int = 200
) -> str:
    text = raw_fields.get(m.source_field, "")
    if not text:
        return ""
    a = max(0, m.start - window_chars)
    b = min(len(text), m.end + window_chars)
    return text[a:b]


@dataclass(frozen=True)
class LinkConfig:
    policy: PolicyConfig = PolicyConfig()
    include_negation: bool = False
    negation: NegationConfig = NegationConfig()
    context_window_chars: int = 200


def link_mentions(
    raw_fields: Dict[str, str],
    mentions: Sequence[Mention],
    ontology_name: str,
    retriever: CandidateRetriever,
    reranker: BaseReranker,
    cfg: LinkConfig,
) -> List[LinkedEntity]:
    out: List[LinkedEntity] = []

    for m in mentions:
        # Optional negation => REJECTED
        if cfg.include_negation and is_negated(raw_fields, m, cfg.negation):
            out.append(
                LinkedEntity(
                    label=m.label,
                    surface_form=m.surface_form,
                    source_field=m.source_field,
                    offsets=m.offsets(),
                    status=EntityStatus.REJECTED,
                    linked_id=None,
                    score=None,
                    margin=None,
                    top_candidates=[],
                    provenances=[m.offsets()],
                )
            )
            continue

        cands = retriever.retrieve(ontology_name, m.surface_form)

        if not cands:
            out.append(
                LinkedEntity(
                    label=m.label,
                    surface_form=m.surface_form,
                    source_field=m.source_field,
                    offsets=m.offsets(),
                    status=EntityStatus.UNRESOLVED,
                    linked_id=None,
                    score=None,
                    margin=None,
                    top_candidates=[],
                    provenances=[m.offsets()],
                )
            )
            continue

        ctx = local_context(raw_fields, m, window_chars=cfg.context_window_chars)
        rr = reranker.rerank(m.surface_form, ctx, cands)

        best = rr.best
        best_score = rr.best_score
        margin = rr.margin
        status = assign_status(best_score, margin, cfg.policy.tau, cfg.policy.delta)

        top_keep = (
            rr.top[: cfg.policy.top_n]
            if status in {EntityStatus.AMBIGUOUS, EntityStatus.UNRESOLVED}
            else rr.top[:1]
        )

        out.append(
            LinkedEntity(
                label=m.label,
                surface_form=m.surface_form,
                source_field=m.source_field,
                offsets=m.offsets(),
                status=status,
                linked_id=(
                    best.candidate_id
                    if status == EntityStatus.RESOLVED and best
                    else None
                ),
                score=float(best_score) if best else None,
                margin=float(margin) if best else None,
                top_candidates=top_keep,
                provenances=[m.offsets()],
            )
        )

    return dedup_entities(out)


def dedup_entities(entities: Sequence[LinkedEntity]) -> List[LinkedEntity]:
    """
    LFR-16:
      - resolved: merge by (label, linked_id)
      - others: merge by (label, normalized surface_form)
    Retain multiple provenance entries.
    Deterministic: stable order by first occurrence.
    """
    merged: Dict[Tuple[str, str], LinkedEntity] = {}
    order: List[Tuple[str, str]] = []

    def key(e: LinkedEntity) -> Tuple[str, str]:
        if e.status == EntityStatus.RESOLVED and e.linked_id:
            return (e.label, f"ID::{e.linked_id}")
        return (e.label, f"SF::{normalize_text(e.surface_form)}")

    for e in entities:
        k = key(e)
        if k not in merged:
            merged[k] = e
            order.append(k)
            continue

        base = merged[k]
        # merge provenances (dedup)
        prov = list(base.provenances)
        for p in e.provenances:
            if p not in prov:
                prov.append(p)

        # Keep the "stronger" status deterministically:
        # RESOLVED > AMBIGUOUS > UNRESOLVED > REJECTED (you can adjust)
        rank = {
            EntityStatus.RESOLVED: 3,
            EntityStatus.AMBIGUOUS: 2,
            EntityStatus.UNRESOLVED: 1,
            EntityStatus.REJECTED: 0,
        }
        chosen = base if rank[base.status] >= rank[e.status] else e
        chosen = LinkedEntity(**{**chosen.model_dump(), "provenances": prov})
        merged[k] = chosen

    return [merged[k] for k in order]
