from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .contracts import Candidate


@dataclass(frozen=True)
class RerankResult:
    best: Candidate | None
    best_score: float
    margin: float
    top: List[Candidate]  # sorted desc by rerank score


class BaseReranker:
    def rerank(
        self, mention_text: str, context: str, candidates: Sequence[Candidate]
    ) -> RerankResult:
        raise NotImplementedError


class DummyReranker(BaseReranker):
    """
    Deterministic reranker for tests:
    - score = lexical score if present else 0.5
    - tie-break by candidate_id
    """

    def rerank(
        self, mention_text: str, context: str, candidates: Sequence[Candidate]
    ) -> RerankResult:
        scored = []
        for c in candidates:
            s = float(c.score) if c.score is not None else 0.5
            scored.append((s, c))

        scored.sort(key=lambda x: (-x[0], x[1].candidate_id))
        top = []
        for s, c in scored:
            top.append(
                Candidate(
                    **{
                        **c.model_dump(),
                        "score": float(s),
                        "source": (c.source or "rerank"),
                    }
                )
            )
        best = top[0] if top else None
        best_score = float(best.score) if best else 0.0
        margin = (
            float(best_score - float(top[1].score))
            if len(top) > 1
            else (1.0 if best else 0.0)
        )
        return RerankResult(best=best, best_score=best_score, margin=margin, top=top)


class CrossEncoderReranker(BaseReranker):
    """
    Uses sentence-transformers CrossEncoder.
    For maximum determinism, prefer device='cpu' in paper runs.
    """

    def __init__(
        self, model_id: str, device: str = "cpu", batch_size: int = 32, seed: int = 0
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import random
            import numpy as np
            import torch

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except Exception:
            pass

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_id, device=self.device)

    def rerank(
        self, mention_text: str, context: str, candidates: Sequence[Candidate]
    ) -> RerankResult:
        self._ensure_loaded()
        assert self._model is not None

        if not candidates:
            return RerankResult(best=None, best_score=0.0, margin=0.0, top=[])

        query = f"{mention_text}\n\nCONTEXT:\n{context}".strip()

        pairs = []
        for c in candidates:
            rhs = c.candidate_label
            if c.definition:
                rhs = f"{rhs}\n\nDEF:\n{c.definition}"
            pairs.append((query, rhs))

        scores = self._model.predict(pairs, batch_size=self.batch_size)
        scored = list(zip([float(x) for x in scores], candidates))
        scored.sort(key=lambda x: (-x[0], x[1].candidate_id))

        top = []
        for s, c in scored:
            top.append(
                Candidate(**{**c.model_dump(), "score": float(s), "source": "rerank"})
            )

        best = top[0]
        best_score = float(best.score)
        margin = float(best_score - float(top[1].score)) if len(top) > 1 else 1.0
        return RerankResult(best=best, best_score=best_score, margin=margin, top=top)
