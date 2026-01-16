from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _gse_raw_dir(layout, gse_id: str) -> Path:
    return layout.run_root / "raw" / gse_id


def load_gse_gsms_raw(*, cfg, layout, gse_id: str) -> List[Dict]:
    """
    Returns:
      [
        {"gsm_id": "...", "raw_fields": {"title": "...", "summary": "...", ...}},
        ...
      ]
    Contract: raw cache must already exist for offline experiments.
    """
    gse_dir = _gse_raw_dir(layout, gse_id)
    if not gse_dir.exists():
        # For experiment integrity, fail fast rather than silently yielding 0 GSM.
        raise FileNotFoundError(f"Missing raw cache for {gse_id}: {gse_dir}")

    gsm_dir = gse_dir / "gsm"
    if not gsm_dir.exists():
        raise FileNotFoundError(f"Missing GSM directory for {gse_id}: {gsm_dir}")

    out: List[Dict] = []
    for p in sorted(gsm_dir.glob("*.json")):
        payload = json.loads(p.read_text("utf-8"))
        out.append({"gsm_id": payload["gsm_id"], "raw_fields": payload["raw_fields"]})
    return out
