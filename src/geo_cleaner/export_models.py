from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .contracts import LinkedEntity
from .textview import SelectedField


SCHEMA_VERSION = "1.0"


class GSMCleanedRecord(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION

    gse_id: str = Field(..., min_length=1)
    gsm_id: str = Field(..., min_length=1)

    # Paper-friendly: store hash always; fields optionally.
    textview_hash: str = Field(..., min_length=1)
    textview_fields: Optional[List[SelectedField]] = None

    # label -> list[LinkedEntity]
    entities: Dict[str, List[LinkedEntity]] = Field(default_factory=dict)


class YieldStats(BaseModel):
    total: int = 0
    resolved: int = 0
    ambiguous: int = 0
    unresolved: int = 0
    rejected: int = 0


class SurfaceFormCount(BaseModel):
    label: str
    source_field: str
    surface_form: str
    count: int


class GSESummary(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION

    gse_id: str = Field(..., min_length=1)
    n_gsm_processed: int = Field(..., ge=0)

    yields_by_label: Dict[str, YieldStats] = Field(default_factory=dict)

    # Top error strings for error analysis
    top_ambiguous: List[SurfaceFormCount] = Field(default_factory=list)
    top_unresolved: List[SurfaceFormCount] = Field(default_factory=list)


class CorpusReport(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION

    run_id: str = Field(..., min_length=1)
    query_terms: List[str] = Field(default_factory=list)
    query_filters: dict = Field(default_factory=dict)

    manifest_path: str = Field(..., min_length=1)
    corpus_gse_ids_path: str = Field(..., min_length=1)

    n_gse_selected: int = Field(..., ge=0)
    n_gse_processed: int = Field(..., ge=0)

    gse_selected: List[str] = Field(default_factory=list)
    gse_processed: List[str] = Field(default_factory=list)

    aggregate_yields_by_label: Dict[str, YieldStats] = Field(default_factory=dict)

    global_top_ambiguous: List[SurfaceFormCount] = Field(default_factory=list)
    global_top_unresolved: List[SurfaceFormCount] = Field(default_factory=list)

    # Optional: helpful to bind results to exact resources in a paper
    resources: dict = Field(default_factory=dict)
