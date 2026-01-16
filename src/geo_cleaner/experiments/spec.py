from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VariantSpec(BaseModel):
    variant_id: str = Field(..., min_length=1)
    overrides: Dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    experiment_id: str = Field(..., min_length=1)
    description: Optional[str] = None

    # Optional: a shared override applied before each variant override
    defaults: Dict[str, Any] = Field(default_factory=dict)

    # Variants to run
    variants: List[VariantSpec] = Field(default_factory=list)

    # Which consolidated artifact(s) to write
    export: Dict[str, Any] = Field(default_factory=dict)

    # Optional: baseline for “delta” calculations
    baseline_variant_id: Optional[str] = None
