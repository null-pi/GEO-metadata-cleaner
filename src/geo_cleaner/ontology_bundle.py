from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contracts import Candidate

_QUOTED_RX = re.compile(r"\"([^\"]+)\"")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # keep alnum, convert everything else to space
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    s = " ".join(s.split())
    return s


def _extract_quoted(text: str) -> str:
    m = _QUOTED_RX.search(text or "")
    return m.group(1) if m else (text or "").strip()


@dataclass(frozen=True)
class Concept:
    concept_id: str
    label: str
    synonyms: Tuple[str, ...] = ()
    definition: Optional[str] = None


class OntologyStore:
    """
    Normalized ontology store:
      - concepts: id -> Concept(label, synonyms, definition?)
      - lexical maps: exact + normalized (string -> list[concept_id])
      - stable version_id: sha256(file bytes)
    """

    def __init__(self, name: str, obo_path: Path):
        self.name = name
        self.obo_path = obo_path
        self.version_hash = file_sha256(obo_path)
        self.version_id = f"{name}:{self.version_hash[:12]}"

        self.concepts: Dict[str, Concept] = {}
        self.lexical_exact: Dict[str, List[str]] = {}
        self.lexical_norm: Dict[str, List[str]] = {}

    def add_concept(self, c: Concept) -> None:
        self.concepts[c.concept_id] = c
        for term in [c.label, *c.synonyms]:
            if not term:
                continue
            self.lexical_exact.setdefault(term, []).append(c.concept_id)
            n = normalize_text(term)
            if n:
                self.lexical_norm.setdefault(n, []).append(c.concept_id)

    def lookup_exact(self, s: str) -> List[str]:
        return list(dict.fromkeys(self.lexical_exact.get(s, [])))

    def lookup_normalized(self, s: str) -> List[str]:
        return list(dict.fromkeys(self.lexical_norm.get(normalize_text(s), [])))


class Embedder:
    """
    Small abstraction to enable fast, deterministic tests.
    Production implementation uses sentence-transformers.
    """

    def model_id(self) -> str:
        raise NotImplementedError

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self, model_name_or_path: str, device: str = "cpu", batch_size: int = 64
    ):
        from sentence_transformers import SentenceTransformer

        self._model_name_or_path = model_name_or_path
        self._device = device
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name_or_path, device=device)

    def model_id(self) -> str:
        return self._model_name_or_path

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # normalize_embeddings=True gives cosine-ready vectors
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


def _faiss_import():
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS import failed. Ensure faiss (faiss-cpu or faiss-gpu) is installed and functional."
        ) from e
    return faiss


@dataclass(frozen=True)
class VectorIndexHandle:
    index_dir: Path
    meta_path: Path
    index_path: Path
    concept_ids_path: Path
    embeddings_path: Path
    reused: bool
    dim: int
    count: int


class OntologyBundle:
    """
    Bundle of multiple ontologies with uniform access + persistent vector indexes.
    """

    def __init__(self, stores: Mapping[str, OntologyStore], cache_dir: Path):
        self.stores = dict(stores)
        self.cache_dir = cache_dir

    @staticmethod
    def load_from_obo_files(
        name_to_path: Mapping[str, Path], cache_dir: Path
    ) -> "OntologyBundle":
        # Heavy deps are isolated here
        import obonet  # type: ignore

        stores: Dict[str, OntologyStore] = {}
        for name, path in name_to_path.items():
            store = OntologyStore(name=name, obo_path=path)

            graph = obonet.read_obo(str(path))
            for node_id, data in graph.nodes(data=True):
                # Filter out non-terms if needed; keep simple: accept nodes with "name"
                label = data.get("name")
                if not isinstance(label, str) or not label.strip():
                    continue

                syns: List[str] = []
                raw_syn = data.get("synonym")
                if isinstance(raw_syn, list):
                    for s in raw_syn:
                        if isinstance(s, str):
                            syns.append(_extract_quoted(s))
                elif isinstance(raw_syn, str):
                    syns.append(_extract_quoted(raw_syn))

                definition: Optional[str] = None
                raw_def = data.get("def")
                if isinstance(raw_def, list) and raw_def:
                    definition = _extract_quoted(str(raw_def[0]))
                elif isinstance(raw_def, str):
                    definition = _extract_quoted(raw_def)

                c = Concept(
                    concept_id=str(node_id),
                    label=label.strip(),
                    synonyms=tuple(x for x in syns if x),
                    definition=definition,
                )
                store.add_concept(c)

            stores[name] = store

        cache_dir.mkdir(parents=True, exist_ok=True)
        return OntologyBundle(stores=stores, cache_dir=cache_dir)

    def version_id(self) -> str:
        """
        Stable bundle id derived from per-ontology version hashes.
        """
        payload = {k: v.version_hash for k, v in sorted(self.stores.items())}
        b = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    def lexical_lookup(self, ontology_name: str, mention: str) -> Dict[str, List[str]]:
        store = self.stores[ontology_name]
        return {
            "exact": store.lookup_exact(mention),
            "normalized": store.lookup_normalized(mention),
        }

    def _index_dir(self, ontology_name: str, embedding_model_id: str) -> Path:
        store = self.stores[ontology_name]
        # Keyed by (ontology_version, embedding_model_id)
        safe_model = normalize_text(embedding_model_id).replace(" ", "_") or "model"
        return (
            self.cache_dir
            / "vector_indexes"
            / ontology_name
            / store.version_hash
            / safe_model
        )

    def get_or_build_vector_index(
        self,
        ontology_name: str,
        embedder: Embedder,
        force_rebuild: bool = False,
    ) -> VectorIndexHandle:
        faiss = _faiss_import()

        store = self.stores[ontology_name]
        index_dir = self._index_dir(ontology_name, embedder.model_id())
        index_dir.mkdir(parents=True, exist_ok=True)

        meta_path = index_dir / "meta.json"
        index_path = index_dir / "faiss.index"
        concept_ids_path = index_dir / "concept_ids.json"
        embeddings_path = index_dir / "embeddings.npy"

        # Reuse if meta matches and files exist
        if (
            not force_rebuild
            and meta_path.exists()
            and index_path.exists()
            and concept_ids_path.exists()
            and embeddings_path.exists()
        ):
            meta = json.loads(meta_path.read_text("utf-8"))
            if (
                meta.get("ontology_version_hash") == store.version_hash
                and meta.get("embedding_model_id") == embedder.model_id()
            ):
                dim = int(meta["dim"])
                count = int(meta["count"])
                return VectorIndexHandle(
                    index_dir=index_dir,
                    meta_path=meta_path,
                    index_path=index_path,
                    concept_ids_path=concept_ids_path,
                    embeddings_path=embeddings_path,
                    reused=True,
                    dim=dim,
                    count=count,
                )

        # Build embeddings (label-only to keep it compact; synonyms handled lexically)
        concept_ids = sorted(store.concepts.keys())
        texts = [store.concepts[cid].label for cid in concept_ids]

        vecs = embedder.encode(texts)
        if vecs.ndim != 2:
            raise ValueError("Embedder must return a 2D array [N, D].")
        vecs = vecs.astype(np.float32)

        # Ensure cosine-ready normalization if embedder didn't do it
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        dim = int(vecs.shape[1])
        count = int(vecs.shape[0])

        # Build FAISS cosine index via inner-product on normalized vectors
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        # Persist artifacts
        np.save(embeddings_path, vecs)
        concept_ids_path.write_text(
            json.dumps(concept_ids, indent=2, sort_keys=True), encoding="utf-8"
        )
        faiss.write_index(index, str(index_path))

        meta = {
            "ontology_name": ontology_name,
            "ontology_version_hash": store.version_hash,
            "ontology_version_id": store.version_id,
            "embedding_model_id": embedder.model_id(),
            "dim": dim,
            "count": count,
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )

        return VectorIndexHandle(
            index_dir=index_dir,
            meta_path=meta_path,
            index_path=index_path,
            concept_ids_path=concept_ids_path,
            embeddings_path=embeddings_path,
            reused=False,
            dim=dim,
            count=count,
        )

    def vector_search(
        self,
        ontology_name: str,
        embedder: Embedder,
        query: str,
        top_k: int = 10,
    ) -> List[Candidate]:
        faiss = _faiss_import()

        handle = self.get_or_build_vector_index(ontology_name, embedder)
        index = faiss.read_index(str(handle.index_path))
        concept_ids = json.loads(handle.concept_ids_path.read_text("utf-8"))

        qv = embedder.encode([query]).astype(np.float32)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

        scores, idxs = index.search(qv, top_k)
        out: List[Candidate] = []
        store = self.stores[ontology_name]
        for j, score in zip(idxs[0].tolist(), scores[0].tolist()):
            if j < 0 or j >= len(concept_ids):
                continue
            cid = concept_ids[j]
            out.append(
                Candidate(
                    candidate_id=cid,
                    candidate_label=store.concepts[cid].label,
                    score=float(score),
                )
            )
        return out
