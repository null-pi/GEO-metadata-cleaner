from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contracts import EntityStatus, LinkedEntity
from .export_models import (
    CorpusReport,
    GSMCleanedRecord,
    GSESummary,
    SurfaceFormCount,
    YieldStats,
)
from .stable_json import stable_dumps


STATUS_RANK = {
    EntityStatus.RESOLVED: 3,
    EntityStatus.AMBIGUOUS: 2,
    EntityStatus.UNRESOLVED: 1,
    EntityStatus.REJECTED: 0,
}


def _sort_candidates_in_entity(e: LinkedEntity) -> LinkedEntity:
    # Ensure candidates have deterministic ordering
    cands = list(e.top_candidates or [])
    cands.sort(key=lambda c: (-float(c.score), c.candidate_id))
    payload = e.model_dump()
    payload["top_candidates"] = [c.model_dump() for c in cands]
    return LinkedEntity.model_validate(payload)


def _sort_entities(entities: List[LinkedEntity]) -> List[LinkedEntity]:
    # Stable order for JSON output (bitwise stability)
    def key(e: LinkedEntity):
        return (
            -STATUS_RANK[e.status],
            e.linked_id or "",
            e.source_field,
            e.offsets.start,
            e.offsets.end,
            e.surface_form,
        )

    normed = [_sort_candidates_in_entity(e) for e in entities]
    return sorted(normed, key=key)


def group_entities_by_label(
    entities: Sequence[LinkedEntity],
) -> Dict[str, List[LinkedEntity]]:
    by: Dict[str, List[LinkedEntity]] = defaultdict(list)
    for e in entities:
        by[e.label].append(e)

    # Stable label ordering and entity ordering
    out: Dict[str, List[LinkedEntity]] = {}
    for lbl in sorted(by.keys()):
        out[lbl] = _sort_entities(by[lbl])
    return out


def write_gsm_jsonl(records: Sequence[GSMCleanedRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(stable_dumps(r.model_dump(mode="json")))
            f.write("\n")


def _yield_stats_from_entities(entities: Iterable[LinkedEntity]) -> YieldStats:
    ys = YieldStats()
    for e in entities:
        ys.total += 1
        if e.status == EntityStatus.RESOLVED:
            ys.resolved += 1
        elif e.status == EntityStatus.AMBIGUOUS:
            ys.ambiguous += 1
        elif e.status == EntityStatus.UNRESOLVED:
            ys.unresolved += 1
        elif e.status == EntityStatus.REJECTED:
            ys.rejected += 1
    return ys


def _top_surface_forms(
    entities: Iterable[LinkedEntity],
    statuses: set[EntityStatus],
    top_n: int,
) -> List[SurfaceFormCount]:
    counts: Counter[Tuple[str, str, str]] = Counter()
    for e in entities:
        if e.status not in statuses:
            continue
        counts[(e.label, e.source_field, e.surface_form)] += 1

    items = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]),
    )
    out: List[SurfaceFormCount] = []
    for (label, source_field, surface_form), cnt in items[:top_n]:
        out.append(
            SurfaceFormCount(
                label=label,
                source_field=source_field,
                surface_form=surface_form,
                count=int(cnt),
            )
        )
    return out


def compute_gse_summary(
    gse_id: str,
    gsm_records: Sequence[GSMCleanedRecord],
    top_n: int = 20,
) -> GSESummary:
    all_entities: List[LinkedEntity] = []
    yields_by_label: Dict[str, YieldStats] = {}

    for r in gsm_records:
        for lbl, ents in r.entities.items():
            all_entities.extend(ents)
            yields_by_label.setdefault(lbl, YieldStats())

    # aggregate yields per label
    for lbl in sorted(yields_by_label.keys()):
        ents = []
        for r in gsm_records:
            ents.extend(r.entities.get(lbl, []))
        yields_by_label[lbl] = _yield_stats_from_entities(ents)

    return GSESummary(
        gse_id=gse_id,
        n_gsm_processed=len(gsm_records),
        yields_by_label=yields_by_label,
        top_ambiguous=_top_surface_forms(
            all_entities, {EntityStatus.AMBIGUOUS}, top_n=top_n
        ),
        top_unresolved=_top_surface_forms(
            all_entities, {EntityStatus.UNRESOLVED}, top_n=top_n
        ),
    )


def compute_corpus_report(
    *,
    run_id: str,
    query_terms: list[str],
    query_filters: dict,
    manifest_path: str,
    corpus_gse_ids_path: str,
    gse_selected: list[str],
    gse_to_summary: Mapping[str, GSESummary],
    top_n: int = 50,
    resources: Optional[dict] = None,
) -> CorpusReport:
    gse_processed = sorted(gse_to_summary.keys())

    # Aggregate yields by label across GSE summaries
    agg: Dict[str, YieldStats] = {}

    all_entities_for_errors: List[Tuple[str, SurfaceFormCount]] = []
    # We need global top ambiguous/unresolved. Easiest: rebuild from summaries’ top lists
    # but better: aggregate from all GSM outputs in a real pipeline. Here, we aggregate from per-GSE top lists.
    # For paper-grade runs, prefer aggregating from GSM JSONL (exact).
    amb_counter: Counter[Tuple[str, str, str]] = Counter()
    unr_counter: Counter[Tuple[str, str, str]] = Counter()

    for s in gse_to_summary.values():
        for lbl, ys in s.yields_by_label.items():
            if lbl not in agg:
                agg[lbl] = YieldStats()
            agg[lbl].total += ys.total
            agg[lbl].resolved += ys.resolved
            agg[lbl].ambiguous += ys.ambiguous
            agg[lbl].unresolved += ys.unresolved
            agg[lbl].rejected += ys.rejected

        for x in s.top_ambiguous:
            amb_counter[(x.label, x.source_field, x.surface_form)] += x.count
        for x in s.top_unresolved:
            unr_counter[(x.label, x.source_field, x.surface_form)] += x.count

    def _counter_to_top(
        counter: Counter[Tuple[str, str, str]], n: int
    ) -> List[SurfaceFormCount]:
        items = sorted(
            counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2])
        )
        out: List[SurfaceFormCount] = []
        for (lbl, field, sf), cnt in items[:n]:
            out.append(
                SurfaceFormCount(
                    label=lbl, source_field=field, surface_form=sf, count=int(cnt)
                )
            )
        return out

    return CorpusReport(
        run_id=run_id,
        query_terms=list(query_terms),
        query_filters=dict(query_filters),
        manifest_path=manifest_path,
        corpus_gse_ids_path=corpus_gse_ids_path,
        n_gse_selected=len(gse_selected),
        n_gse_processed=len(gse_processed),
        gse_selected=list(gse_selected),
        gse_processed=gse_processed,
        aggregate_yields_by_label={k: agg[k] for k in sorted(agg.keys())},
        global_top_ambiguous=_counter_to_top(amb_counter, top_n),
        global_top_unresolved=_counter_to_top(unr_counter, top_n),
        resources=resources or {},
    )


def write_json(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_dumps(model.model_dump(mode="json")), encoding="utf-8")
