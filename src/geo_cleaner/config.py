from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class NCBIConfig(BaseModel):
    """Configuration for NCBI Entrez utilities.

    Attributes:
        base_url: Base URL for NCBI Entrez API.
        tool: Name of the tool using the API.
        email: Contact email address.
        api_key: Optional API key for NCBI.
        timeout_s: Timeout for API requests in seconds.
        rps: Requests per second rate limit.
    """
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    tool: str = "geo_cleaner"
    email: str = "unknown@example.com"
    api_key: Optional[str] = None
    timeout_s: float = 30.0
    rps: float = 3.0


class RunConfig(BaseModel):
    """Configuration for a run, including output directory and NCBI settings.

    Attributes:
        out_dir: Output directory for run artifacts.
        ncbi: NCBI configuration.
    """
    out_dir: Path = Path("runs")
    ncbi: NCBIConfig = Field(default_factory=NCBIConfig)


class QueryFilters(BaseModel):
    """Filters for querying GEO datasets.

    Attributes:
        organism: Organism name filter.
        date_start: Start date filter (YYYY-MM-DD).
        date_end: End date filter (YYYY-MM-DD).
        max_gse: Maximum number of GSE records to retrieve.
    """
    organism: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    max_gse: int = 200


class TextViewConfig(BaseModel):
    """Configuration for text view and field prioritization.

    Attributes:
        field_priority: List of field names in order of priority.
        max_field_chars: Maximum number of characters per field for display.
    """
    field_priority: list[str] = Field(
        default_factory=lambda: [
            "title",
            "summary",
            "overall_design",
            "type",
            "organism_ch1",
            "characteristics_ch1",
            "source_name_ch1",
            "treatment_protocol_ch1",
            "growth_protocol_ch1",
            "extract_protocol_ch1",
            "library_strategy",
            "platform_id",
        ]
    )
    # Optional: clip huge fields for model context without losing raw evidence elsewhere
    max_field_chars: int = 4000


class CanonicalConfig(BaseModel):
    """Top-level configuration model for the GEO metadata cleaner.

    Attributes:
        run: Run configuration.
        query: Query filters.
        models: Placeholder for model configurations.
        ontologies: Placeholder for ontology configurations.
        textview: Text view configuration.
    """
    run: RunConfig = Field(default_factory=RunConfig)
    query: QueryFilters = Field(default_factory=QueryFilters)
    # Placeholders for later epics; kept for manifest completeness
    models: dict[str, Any] = Field(default_factory=dict)
    ontologies: dict[str, Any] = Field(default_factory=dict)
    textview: TextViewConfig = Field(default_factory=TextViewConfig)


def _canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object to canonical JSON bytes.

    Args:
        obj: The object to serialize.

    Returns:
        Canonical JSON bytes with sorted keys and no whitespace.
    """
    # Stable serialization: sort keys, no whitespace
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def config_hash(cfg: CanonicalConfig) -> str:
    """Compute a SHA-256 hash of the canonical configuration.

    Args:
        cfg: The CanonicalConfig instance.

    Returns:
        A SHA-256 hexadecimal hash string of the configuration.
    """
    payload = cfg.model_dump(mode="json")
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def load_config(path: Optional[Path]) -> CanonicalConfig:
    """Load configuration from a YAML or JSON file.

    Args:
        path: Path to the configuration file. If None, returns default config.

    Returns:
        An instance of CanonicalConfig loaded from the file or defaults.

    Raises:
        ValueError: If the file format is unsupported.
    """
    if path is None:
        # Return default configuration if no path is provided
        return CanonicalConfig()

    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(raw) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(f"Unsupported config format: {path}")

    return CanonicalConfig.model_validate(data)
