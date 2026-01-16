from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .extractor import ExtractorConfig, GLiNERExtractor, RegexExtractor
from .linker import LinkConfig, link_mentions
from .candidate_retrieval import CandidateRetriever
from .reranker import BaseReranker
from .status_policy import PolicyConfig
from .export_models import GSMCleanedRecord
from .exporter import (
    compute_corpus_report,
    compute_gse_summary,
    group_entities_by_label,
    write_gsm_jsonl,
)
from .stable_json import stable_dumps


# ---------------------------
# Public result contract
# ---------------------------


@dataclass(frozen=True)
class PipelineStats:
    gse_processed: int
    gsm_processed: int
    entities_total: int


# ---------------------------
# Small helpers
# ---------------------------


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(stable_dumps(obj), encoding="utf-8")


def _iter_gsm_cache_files(gse_root: Path) -> List[Path]:
    """
    Supports either:
      raw/<GSE>/gsm/<GSM>.json   (preferred)
    or
      raw/<GSE>/gsms.json        (alternative)
    """
    gsm_dir = gse_root / "gsm"
    if gsm_dir.exists():
        files = sorted(gsm_dir.glob("*.json"))
        if files:
            return files

    gsms_json = gse_root / "gsms.json"
    if gsms_json.exists():
        return [gsms_json]

    return []


def _load_gsms_from_cache(gse_root: Path) -> List[Dict[str, Any]]:
    files = _iter_gsm_cache_files(gse_root)
    if not files:
        return []

    # Case A: many GSM json files
    if len(files) > 1 or files[0].name != "gsms.json":
        out: List[Dict[str, Any]] = []
        for p in files:
            payload = _safe_read_json(p)
            # Expect: {"gsm_id": "...", "raw_fields": {...}}
            if (
                isinstance(payload, dict)
                and "gsm_id" in payload
                and "raw_fields" in payload
            ):
                out.append(payload)
        return out

    # Case B: a single gsms.json file holding a list
    payload = _safe_read_json(files[0])
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if (
        isinstance(payload, dict)
        and "gsms" in payload
        and isinstance(payload["gsms"], list)
    ):
        return [x for x in payload["gsms"] if isinstance(x, dict)]
    return []


def _copy_gse_raw_cache(*, src_raw: Path, dst_raw: Path, gse_id: str) -> bool:
    src = src_raw / gse_id
    dst = dst_raw / gse_id
    if not src.exists():
        return False
    if dst.exists():
        return True
    _ensure_dir(dst.parent)
    shutil.copytree(src, dst)
    return True


def _default_label_to_ontology() -> Dict[str, str]:
    # Safe defaults; you can later move this into config.
    return {
        "disease": "doid",
        "tissue": "uberon",
        "organism": "ncbitaxon",
        "cell_type": "cl",
        "cell line": "cellosaurus",
        "cell_line": "cellosaurus",
        "drug": "chebi",
        "assay": "efo",
        "platform": "efo",
    }


def _build_textview_payload(raw_fields: Mapping[str, str], cfg: Any) -> Dict[str, Any]:
    """
    Deterministic TextView. We store:
      - fields_selected
      - concatenated_text (optional; included if your GSMCleanedRecord schema wants it)
      - hash
    If your project already has textview.py, you can replace this with it later.
    """
    # Best-effort: read priority list from cfg.textview.priority if present.
    priority: List[str] = []
    try:
        tv = getattr(cfg, "textview", None)
        if tv is not None:
            priority = list(getattr(tv, "priority", []) or [])
    except Exception:
        priority = []

    if not priority:
        priority = [
            "title",
            "source_name_ch1",
            "characteristics_ch1",
            "summary",
            "description",
        ]

    fields_selected: List[str] = []
    parts: List[str] = []
    for k in priority:
        v = raw_fields.get(k, "")
        if isinstance(v, str) and v.strip():
            fields_selected.append(k)
            parts.append(f"{k}: {v.strip()}")

    concatenated = "\n\n".join(parts)
    h = _sha256_text(concatenated)

    return {
        "fields_selected": fields_selected,
        "concatenated_text": concatenated,
        "hash": h,
    }


def _make_gsm_record(
    *,
    gse_id: str,
    gsm_id: str,
    raw_fields: Dict[str, str],
    entities_by_label,
    cfg: Any,
) -> GSMCleanedRecord:
    """
    Builds a GSMCleanedRecord in a schema-tolerant way by introspecting model fields.
    This keeps the pipeline robust even if you slightly rename fields in export_models.py.
    """
    model_fields = set(getattr(GSMCleanedRecord, "model_fields", {}).keys())

    tv = _build_textview_payload(raw_fields, cfg)

    payload: Dict[str, Any] = {}
    if "gse_id" in model_fields:
        payload["gse_id"] = gse_id
    if "gsm_id" in model_fields:
        payload["gsm_id"] = gsm_id

    # TextView variants
    if "textview_hash" in model_fields:
        payload["textview_hash"] = tv["hash"]
    if "textview_fields" in model_fields:
        payload["textview_fields"] = tv["fields_selected"]
    if "fields_selected" in model_fields:
        payload["fields_selected"] = tv["fields_selected"]

    if "textview" in model_fields:
        # Try to match the expected TextView model keys.
        tv_obj: Dict[str, Any] = {}
        # Prefer minimal paper-friendly fields; include concatenated_text only if likely expected.
        tv_obj["hash"] = tv["hash"]
        tv_obj["fields_selected"] = tv["fields_selected"]
        tv_obj["concatenated_text"] = tv["concatenated_text"]
        payload["textview"] = tv_obj

    # Entities
    payload["entities"] = entities_by_label

    return GSMCleanedRecord.model_validate(payload)


def _build_extractor(cfg: Any):
    """
    Builds a high-recall extractor.
    - If cfg.models.extractor (or cfg.models.gliner) is set: GLiNERExtractor
    - Else: RegexExtractor (only for smoke checks; will extract nothing unless configured)
    """
    model_id: Optional[str] = None
    labels: Tuple[str, ...] = (
        "disease",
        "tissue",
        "organism",
        "cell_type",
        "drug",
        "cell_line",
        "assay",
    )

    # Read labels if present
    try:
        ex = getattr(cfg, "extractor", None)
        if ex is not None:
            labels = tuple(getattr(ex, "labels", labels) or labels)
    except Exception:
        pass

    # Read model id
    models = getattr(cfg, "models", None)
    if isinstance(models, dict):
        model_id = models.get("extractor") or models.get("gliner")
    else:
        try:
            model_id = getattr(models, "extractor", None) or getattr(
                models, "gliner", None
            )
        except Exception:
            model_id = None

    device = "cpu"
    try:
        if isinstance(models, dict):
            device = str(models.get("device") or device)
        else:
            device = str(getattr(models, "device", device))
    except Exception:
        device = "cpu"

    if model_id:
        return GLiNERExtractor(
            model_name_or_path=model_id,
            cfg=ExtractorConfig(labels=labels, device=device, threshold=0.0, seed=0),
        )

    # Smoke-only fallback (deterministic, but will not do real NER unless you provide patterns)
    return RegexExtractor(patterns={}, conf=1.0)


def _build_retriever(cfg: Any, cache_dir: Path) -> CandidateRetriever:
    """
    Best-effort builder for CandidateRetriever.
    Your existing CandidateRetriever should support either:
      - CandidateRetriever.from_config(cfg, cache_dir=...)
      - CandidateRetriever(cfg=..., cache_dir=...)
    """
    # classmethod path
    if hasattr(CandidateRetriever, "from_config"):
        return CandidateRetriever.from_config(cfg, cache_dir=cache_dir)  # type: ignore

    # ctor path
    try:
        return CandidateRetriever(cfg=cfg, cache_dir=cache_dir)  # type: ignore
    except TypeError:
        # last resort: try positional (cfg, cache_dir)
        return CandidateRetriever(cfg, cache_dir)  # type: ignore


def _build_reranker(cfg: Any) -> BaseReranker:
    """
    Your reranker module should already implement a concrete BaseReranker.
    We try common factory patterns.
    """
    import geo_cleaner.reranker as rr  # local import to avoid circulars

    if hasattr(rr, "build_reranker"):
        return rr.build_reranker(cfg)  # type: ignore

    # common class names
    for name in ("CrossEncoderReranker", "MiniLMReranker", "Reranker"):
        cls = getattr(rr, name, None)
        if cls is not None:
            try:
                return cls(cfg)  # type: ignore
            except TypeError:
                return cls()  # type: ignore

    raise RuntimeError(
        "Could not construct a reranker. Implement geo_cleaner.reranker.build_reranker(cfg) "
        "or provide a concrete reranker class."
    )


def _build_link_cfg(cfg: Any) -> LinkConfig:
    policy = PolicyConfig()
    include_negation = False
    context_window_chars = 200

    # Try to read from cfg.linker or cfg.link
    try:
        linker = getattr(cfg, "linker", None) or getattr(cfg, "link", None)
        if linker is not None:
            pol = getattr(linker, "policy", None)
            if isinstance(pol, dict):
                policy = PolicyConfig(**pol)
            else:
                # dataclass-like
                tau = getattr(pol, "tau", None)
                delta = getattr(pol, "delta", None)
                top_n = getattr(pol, "top_n", None)
                kwargs = {}
                if tau is not None:
                    kwargs["tau"] = tau
                if delta is not None:
                    kwargs["delta"] = delta
                if top_n is not None:
                    kwargs["top_n"] = top_n
                if kwargs:
                    policy = PolicyConfig(**kwargs)

            include_negation = bool(
                getattr(linker, "include_negation", include_negation)
            )
            context_window_chars = int(
                getattr(linker, "context_window_chars", context_window_chars)
            )
    except Exception:
        pass

    return LinkConfig(
        policy=policy,
        include_negation=include_negation,
        context_window_chars=context_window_chars,
    )


def _load_manifest_bits(layout) -> Tuple[str, List[str], Dict[str, Any], str]:
    m = _safe_read_json(layout.manifest_path)
    run_id = str(m.get("run_id", ""))
    query_terms = list(m.get("query_terms", []) or [])
    query_filters = dict(m.get("query_filters", {}) or {})
    corpus_gse_ids_path = str(
        m.get("corpus_gse_ids_path", str(layout.corpus_gse_ids_path))
    )
    return run_id, query_terms, query_filters, corpus_gse_ids_path


# ---------------------------
# Core pipeline
# ---------------------------


def run_pipeline(
    *,
    cfg: Any,
    layout: Any,
    gse_ids: Sequence[str],
    raw_source_dir: Optional[Path] = None,
) -> PipelineStats:
    """
    Offline-first pipeline:
      - Reuses raw GSM caches from raw_source_dir if provided.
      - Writes:
          outputs/gsm.jsonl
          reports/gse_summary.json
          reports/corpus_report.json
    """
    # dirs
    outputs_dir = layout.run_root / "outputs"
    reports_dir = layout.run_root / "reports"
    raw_dir = layout.run_root / "raw"
    cache_dir = layout.run_root / "cache"
    _ensure_dir(outputs_dir)
    _ensure_dir(reports_dir)
    _ensure_dir(raw_dir)
    _ensure_dir(cache_dir)

    # build components
    extractor = _build_extractor(cfg)
    retriever = _build_retriever(cfg, cache_dir=cache_dir)
    reranker = _build_reranker(cfg)
    link_cfg = _build_link_cfg(cfg)

    label_to_ont = _default_label_to_ontology()

    # Process
    gse_processed = 0
    gsm_processed = 0
    entities_total = 0

    all_gsm_records: List[GSMCleanedRecord] = []
    gse_to_summary = {}

    # deterministic order
    gse_ids_sorted = sorted(set(gse_ids))

    for gse_id in gse_ids_sorted:
        # ensure raw cache exists in this run
        if raw_source_dir is not None:
            _copy_gse_raw_cache(src_raw=raw_source_dir, dst_raw=raw_dir, gse_id=gse_id)

        gse_raw_root = raw_dir / gse_id
        gsms = _load_gsms_from_cache(gse_raw_root)

        if not gsms:
            raise FileNotFoundError(
                f"No raw GSM cache found for {gse_id} under {gse_raw_root}. "
                f"Provide a populated raw cache (e.g., from your corpora run) or add an acquisition stage."
            )

        # deterministic GSM order
        gsms_sorted = sorted(
            gsms,
            key=lambda x: str(x.get("gsm_id", "")),
        )

        gse_records: List[GSMCleanedRecord] = []

        for gsm_payload in gsms_sorted:
            gsm_id = str(gsm_payload.get("gsm_id", "")).strip()
            raw_fields = gsm_payload.get("raw_fields", {})
            if not gsm_id or not isinstance(raw_fields, dict):
                continue

            # normalize raw fields to str
            raw_fields_str: Dict[str, str] = {}
            for k, v in raw_fields.items():
                if v is None:
                    continue
                if isinstance(v, str):
                    raw_fields_str[str(k)] = v
                else:
                    raw_fields_str[str(k)] = str(v)

            # 1) extract mentions (field-scoped offsets)
            mentions = extractor.extract(raw_fields_str)

            # 2) group by label => correct ontology per label
            by_label: Dict[str, List[Any]] = {}
            for m in mentions:
                by_label.setdefault(m.label, []).append(m)

            linked_entities_all = []
            for lbl in sorted(by_label.keys()):
                ont_name = label_to_ont.get(lbl, "doid")
                linked_entities_all.extend(
                    link_mentions(
                        raw_fields=raw_fields_str,
                        mentions=by_label[lbl],
                        ontology_name=ont_name,
                        retriever=retriever,
                        reranker=reranker,
                        cfg=link_cfg,
                    )
                )

            entities_by_label = group_entities_by_label(linked_entities_all)
            entities_total += sum(len(v) for v in entities_by_label.values())

            rec = _make_gsm_record(
                gse_id=gse_id,
                gsm_id=gsm_id,
                raw_fields=raw_fields_str,
                entities_by_label=entities_by_label,
                cfg=cfg,
            )
            gse_records.append(rec)

            gsm_processed += 1

        # per-GSE summary
        gse_summary = compute_gse_summary(gse_id, gse_records, top_n=20)
        gse_to_summary[gse_id] = gse_summary
        all_gsm_records.extend(gse_records)
        gse_processed += 1

    # write gsm.jsonl
    gsm_jsonl_path = outputs_dir / "gsm.jsonl"
    write_gsm_jsonl(all_gsm_records, gsm_jsonl_path)

    # write gse_summary.json (run-level wrapper, stable)
    gse_summary_path = reports_dir / "gse_summary.json"
    gse_summary_obj = {
        "run_id": str(layout.run_root.name),
        "processed_gse_count": gse_processed,
        "processed_gsm_count": gsm_processed,
        "by_gse": {k: v.model_dump(mode="json") for k, v in gse_to_summary.items()},
    }
    _write_json(gse_summary_path, gse_summary_obj)

    # write corpus_report.json (paper artifact)
    run_id, query_terms, query_filters, corpus_gse_ids_path = _load_manifest_bits(
        layout
    )
    corpus_report = compute_corpus_report(
        run_id=run_id or str(layout.run_root.name),
        query_terms=query_terms,
        query_filters=query_filters,
        manifest_path=str(layout.manifest_path),
        corpus_gse_ids_path=corpus_gse_ids_path,
        gse_selected=list(gse_ids_sorted),
        gse_to_summary=gse_to_summary,
        top_n=50,
        resources={
            "models": getattr(cfg, "models", {}) or {},
            "ontologies": getattr(cfg, "ontologies", {}) or {},
        },
    )
    corpus_report_path = reports_dir / "corpus_report.json"
    _write_json(corpus_report_path, corpus_report.model_dump(mode="json"))

    return PipelineStats(
        gse_processed=gse_processed,
        gsm_processed=gsm_processed,
        entities_total=entities_total,
    )
