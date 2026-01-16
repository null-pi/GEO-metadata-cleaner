from pathlib import Path

import numpy as np
import pytest

from geo_cleaner.candidate_retrieval import CandidateRetriever, RetrievalConfig
from geo_cleaner.ontology_bundle import Embedder, OntologyBundle


class DummyEmbedder(Embedder):
    def __init__(self, mid: str = "dummy-embedder-v1", dim: int = 8):
        self._mid = mid
        self._dim = dim

    def model_id(self) -> str:
        return self._mid

    def encode(self, texts):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = sum(ord(c) for c in t) % 997
            out[i] = np.array(
                [(h + k * 13) % 101 for k in range(self._dim)], dtype=np.float32
            )
        return out


@pytest.fixture
def tiny_obo(tmp_path: Path) -> Path:
    p = tmp_path / "toy.obo"
    p.write_text(
        "\n".join(
            [
                "format-version: 1.2",
                "",
                "[Term]",
                "id: TOY:0001",
                "name: Lung cancer",
                'synonym: "lung carcinoma" EXACT []',
                'def: "A malignant tumor of the lung." []',
                "",
                "[Term]",
                "id: TOY:0002",
                "name: Breast cancer",
                'synonym: "mammary carcinoma" EXACT []',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return p


def test_candidate_retrieval_returns_topk_with_scores(tmp_path: Path, tiny_obo: Path):
    bundle = OntologyBundle.load_from_obo_files(
        {"toy": tiny_obo}, cache_dir=tmp_path / "cache"
    )
    retriever = CandidateRetriever(bundle, DummyEmbedder(), RetrievalConfig(top_k=2))

    cands = retriever.retrieve("toy", "lung cancer")
    assert 1 <= len(cands) <= 2
    assert all(isinstance(c.score, float) for c in cands)
    assert all(c.source in {"lexical_exact", "lexical_norm", "vector"} for c in cands)


def test_lexical_candidates_included_when_available(tmp_path: Path, tiny_obo: Path):
    bundle = OntologyBundle.load_from_obo_files(
        {"toy": tiny_obo}, cache_dir=tmp_path / "cache"
    )
    retriever = CandidateRetriever(bundle, DummyEmbedder(), RetrievalConfig(top_k=5))

    cands = retriever.retrieve("toy", "lung carcinoma")
    ids = [c.candidate_id for c in cands]
    assert "TOY:0001" in ids
    assert any(
        c.candidate_id == "TOY:0001" and c.source.startswith("lexical") for c in cands
    )


def test_embedding_retrieval_works_when_lexical_empty(tmp_path: Path, tiny_obo: Path):
    bundle = OntologyBundle.load_from_obo_files(
        {"toy": tiny_obo}, cache_dir=tmp_path / "cache"
    )
    retriever = CandidateRetriever(bundle, DummyEmbedder(), RetrievalConfig(top_k=3))

    # lexical should be empty; vector should still return something
    cands = retriever.retrieve("toy", "lc")  # not a synonym/label
    assert len(cands) > 0
    assert any(c.source == "vector" for c in cands)


def test_caches_reused_across_mentions_and_runs(tmp_path: Path, tiny_obo: Path):
    emb = DummyEmbedder()
    cache = tmp_path / "cache"

    bundle1 = OntologyBundle.load_from_obo_files({"toy": tiny_obo}, cache_dir=cache)
    h1 = bundle1.get_or_build_vector_index("toy", emb, force_rebuild=True)
    assert h1.reused is False

    # "new run": new bundle instance, same cache dir
    bundle2 = OntologyBundle.load_from_obo_files({"toy": tiny_obo}, cache_dir=cache)
    h2 = bundle2.get_or_build_vector_index("toy", emb, force_rebuild=False)
    assert h2.reused is True
