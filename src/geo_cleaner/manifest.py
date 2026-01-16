from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RunManifest:
    run_id: str
    created_at_utc: str

    # reproducibility core
    query_terms: List[str]
    query_filters: Dict[str, Any]
    retrieval_timestamp_utc: str

    code_version: str
    model_ids: Dict[str, Any]
    ontology_versions: Dict[str, Any]
    device: Dict[str, Any]

    # derived outputs
    config_hash: str
    config_path: str
    corpus_gse_ids_path: Optional[str] = None
    corpus_gse_count: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")
