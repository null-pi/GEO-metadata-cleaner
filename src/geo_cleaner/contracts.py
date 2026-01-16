from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EntityStatus(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"
    REJECTED = "REJECTED"


class FieldOffsets(BaseModel):
    field_key: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_span(self):
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class Mention(BaseModel):
    # LFR-7: authoritative evidence = field-scoped offsets into raw field text
    label: str = Field(..., min_length=1)
    surface_form: str = Field(..., min_length=1)
    source_field: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    extractor_conf: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_span(self):
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if self.surface_form == "":
            raise ValueError("surface_form must be non-empty")
        return self

    def offsets(self) -> FieldOffsets:
        return FieldOffsets(field_key=self.source_field, start=self.start, end=self.end)


class Candidate(BaseModel):
    candidate_id: str = Field(..., min_length=1)
    candidate_label: str = Field(..., min_length=1)
    score: float = Field(...)
    source: str | None = None
    definition: str | None = None


class LinkedEntity(BaseModel):
    # LFR-8 minimal provenance
    label: str = Field(..., min_length=1)
    surface_form: str = Field(..., min_length=1)
    source_field: str = Field(..., min_length=1)

    offsets: FieldOffsets
    status: EntityStatus

    linked_id: Optional[str] = None
    score: Optional[float] = None
    margin: Optional[float] = None

    # Required even when unresolved: keep it empty if you have none.
    top_candidates: list[Candidate] = Field(default_factory=list)
    provenances: list[FieldOffsets] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_policy(self):
        if self.status == EntityStatus.RESOLVED and not self.linked_id:
            raise ValueError("linked_id is required when status=RESOLVED")

        if self.offsets.field_key != self.source_field:
            raise ValueError("offsets.field_key must equal source_field")

        # Ensure provenances always contains the primary offsets (first)
        if not self.provenances:
            self.provenances = [self.offsets]
        else:
            if self.offsets not in self.provenances:
                self.provenances.insert(0, self.offsets)

        return self
