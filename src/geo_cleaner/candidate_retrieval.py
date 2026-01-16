from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .contracts import Candidate
from .ontology_bundle import OntologyBundle, normalize_text, Embedder


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    lexical_exact_score: float = 1.0
    lexical_norm_score: float = 0.90
    vector_min_score: float = -1.0  # cosine/IP; keep permissive by default
    include_definitions: bool = True


class CandidateRetriever:
    """
    Retrieval policy:
      - lexical exact + normalized (label/synonyms)
      - vector retrieval as complement/fallback
      - merge by concept_id keeping max score; stable ordering
    """

    def __init__(
        self, bundle: OntologyBundle, embedder: Optional[Embedder], cfg: RetrievalConfig
    ):
        self.bundle = bundle
        self.embedder = embedder
        self.cfg = cfg

    def retrieve(self, ontology_name: str, mention_text: str) -> List[Candidate]:
        store = self.bundle.stores[ontology_name]

        merged: Dict[str, Candidate] = {}

        # 1) Lexical exact / normalized
        lex = self.bundle.lexical_lookup(ontology_name, mention_text)
        for cid in lex["exact"]:
            cobj = store.concepts[cid]
            cand = Candidate(
                candidate_id=cid,
                candidate_label=cobj.label,
                definition=cobj.definition if self.cfg.include_definitions else None,
                score=self.cfg.lexical_exact_score,
                source="lexical_exact",
            )
            merged[cid] = cand

        for cid in lex["normalized"]:
            if cid in merged:
                continue
            cobj = store.concepts[cid]
            cand = Candidate(
                candidate_id=cid,
                candidate_label=cobj.label,
                definition=cobj.definition if self.cfg.include_definitions else None,
                score=self.cfg.lexical_norm_score,
                source="lexical_norm",
            )
            merged[cid] = cand

        # 2) Vector retrieval (complement/fallback)
        if self.embedder is not None:
            vec_cands = self.bundle.vector_search(
                ontology_name,
                self.embedder,
                mention_text,
                top_k=max(self.cfg.top_k, 10),
            )
            for vc in vec_cands:
                if vc.score < self.cfg.vector_min_score:
                    continue
                # enrich with definition
                cobj = store.concepts.get(vc.candidate_id)
                definition = (
                    cobj.definition if (cobj and self.cfg.include_definitions) else None
                )

                if vc.candidate_id in merged:
                    # keep max score; prefer lexical source if lexical already present
                    if vc.score > merged[vc.candidate_id].score:
                        merged[vc.candidate_id].score = float(vc.score)
                    continue

                merged[vc.candidate_id] = Candidate(
                    candidate_id=vc.candidate_id,
                    candidate_label=vc.candidate_label,
                    definition=definition,
                    score=float(vc.score),
                    source="vector",
                )

        # 3) Top-K + stable ordering
        out = sorted(merged.values(), key=lambda c: (-float(c.score), c.candidate_id))
        return out[: self.cfg.top_k]
