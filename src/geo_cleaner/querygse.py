from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from .ncbi_client import NCBIClient


_DATE_RX = re.compile(r"^\d{4}(-\d{2})?(-\d{2})?$")


def _normalize_date_for_pdat(s: str) -> str:
    """
    Accept: YYYY, YYYY-MM, YYYY-MM-DD
    Return: YYYY, YYYY/MM, YYYY/MM/DD
    """
    if not _DATE_RX.match(s):
        # allow passing already formatted values like 2007/01 or 2007/01/15
        return s.replace("-", "/")
    return s.replace("-", "/")


def _maybe_quote(term: str) -> str:
    t = term.strip()
    # If user already wrote advanced syntax, do not touch.
    if any(x in t for x in ["[", "]", "(", ")", " AND ", " OR ", " NOT ", '"']):
        return t
    if " " in t:
        return f'"{t}"'
    return t


@dataclass(frozen=True)
class QueryInputs:
    terms: List[str]
    organism: Optional[str]
    date_start: Optional[str]
    date_end: Optional[str]
    max_gse: int


def build_gds_query(q: QueryInputs, single_term: str) -> str:
    parts: List[str] = []
    parts.append(_maybe_quote(single_term))
    parts.append("gse[ETYP]")

    if q.organism:
        # ORGN is the GEO DataSets organism field alias
        parts.append(f"{_maybe_quote(q.organism)}[ORGN]")

    if q.date_start and q.date_end:
        ds = _normalize_date_for_pdat(q.date_start)
        de = _normalize_date_for_pdat(q.date_end)
        # PDAT range syntax is documented in GEO query examples
        parts.append(f"{ds}:{de}[PDAT]")

    return " AND ".join(
        f"({p})" if " " in p and not p.startswith("(") and p != "gse[ETYP]" else p
        for p in parts
    )


def _extract_accessions_from_esummary_json(payload: Dict[str, Any]) -> List[str]:
    """
    Typical structure: payload["result"]["uids"] and payload["result"][uid]["accession"].
    We defensively scan for any value that looks like GSE\d+ as fallback.
    """
    out: List[str] = []
    result = payload.get("result", {})
    uids = result.get("uids") or []
    for uid in uids:
        rec = result.get(str(uid), {}) or {}
        acc = rec.get("accession")
        if isinstance(acc, str) and acc.startswith("GSE"):
            out.append(acc)
            continue

        # fallback scan
        for v in rec.values():
            if isinstance(v, str) and v.startswith("GSE"):
                out.append(v)
                break
    return out


def query_gse_ids(
    client: NCBIClient, q: QueryInputs
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Returns:
      (deduped_sorted_gse_ids, debug_meta)
    """
    per_term: Dict[str, List[str]] = {}
    union: Set[str] = set()

    for term in q.terms:
        gds_term = build_gds_query(q, term)

        # ESearch in db=gds; retmode=json supported by E-utilities.
        es = client.get(
            "esearch.fcgi",
            {
                "db": "gds",
                "term": gds_term,
                "retmode": "json",
                "retmax": str(max(q.max_gse, 0)),
            },
        ).json()

        idlist = (es.get("esearchresult", {}) or {}).get("idlist", []) or []
        if not idlist:
            per_term[term] = []
            continue

        # ESummary: map UIDs -> accessions (GSE...)
        sm = client.get(
            "esummary.fcgi",
            {
                "db": "gds",
                "id": ",".join(idlist),
                "retmode": "json",
            },
        ).json()

        gse_list = [
            x for x in _extract_accessions_from_esummary_json(sm) if x.startswith("GSE")
        ]
        per_term[term] = gse_list
        union.update(gse_list)

    # Deterministic ordering: numeric sort on the accession suffix.
    def gse_key(x: str) -> int:
        try:
            return int(x.replace("GSE", ""))
        except Exception:
            return 10**18

    merged = sorted(union, key=gse_key)
    if q.max_gse > 0:
        merged = merged[: q.max_gse]

    debug = {"per_term": per_term}
    return merged, debug
