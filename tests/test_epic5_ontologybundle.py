import time
from pathlib import Path

import numpy as np
import pytest

from geo_cleaner.ontology_bundle import Embedder, OntologyBundle, normalize_text


class DummyEmbedder(Embedder):
    def __init__(self, mid: str = "dummy-embedder-v1", dim: int = 8):
        self._mid = mid
        self._dim = dim

    def model_id(self) -> str:
        return self._mid

    def encode(self, texts):
        # deterministic embedding: hash-based
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = sum(ord(c) for c in t) % 997
            # simple, deterministic pseudo-vector
            out[i] = np.array(
                [(h + k * 13) % 101 for k in range(self._dim)], dtype=np.float32
            )
        return out


@pytest.fixture
def tiny_obo(tmp_path: Path) -> Path:
    # minimal OBO with synonyms
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


def test_ontologybundle_exposes_version_id_or_hash(tmp_path: Path, tiny_obo: Path):
    bundle1 = OntologyBundle.load_from_obo_files(
        {"toy": tiny_obo}, cache_dir=tmp_path / "cache"
    )
    bundle2 = OntologyBundle.load_from_obo_files(
        {"toy": tiny_obo}, cache_dir=tmp_path / "cache2"
    )

    assert bundle1.stores["toy"].version_hash == bundle2.stores["toy"].version_hash
    assert bundle1.stores["toy"].version_id == bundle2.stores["toy"].version_id
    assert bundle1.version_id() == bundle2.version_id()


def test_lexical_lookup_exact_and_normalized(tmp_path: Path, tiny_obo: Path):
    bundle = OntologyBundle.load_from_obo_files(
        {"toy": tiny_obo}, cache_dir=tmp_path / "cache"
    )

    # exact label
    r1 = bundle.lexical_lookup("toy", "Lung cancer")
    assert "TOY:0001" in r1["exact"]

    # exact synonym
    r2 = bundle.lexical_lookup("toy", "lung carcinoma")
    assert "TOY:0001" in r2["exact"]

    # normalized variant
    r3 = bundle.lexical_lookup("toy", "  lung   carcinoma!! ")
    assert "TOY:0001" in r3["normalized"]


def test_vector_index_persisted_and_reused(tmp_path: Path, tiny_obo: Path):
    faiss = pytest.importorskip("faiss")  # skip if FAISS is not functional

    cache = tmp_path / "cache"
    bundle = OntologyBundle.load_from_obo_files({"toy": tiny_obo}, cache_dir=cache)
    emb = DummyEmbedder()

    h1 = bundle.get_or_build_vector_index("toy", emb, force_rebuild=True)
    assert h1.reused is False
    assert h1.index_path.exists()
    t1 = h1.index_path.stat().st_mtime

    # Sleep so mtime resolution cannot trick us
    time.sleep(0.2)

    h2 = bundle.get_or_build_vector_index("toy", emb, force_rebuild=False)
    assert h2.reused is True
    t2 = h2.index_path.stat().st_mtime

    # no rebuild => index file unchanged
    assert t2 == t1
