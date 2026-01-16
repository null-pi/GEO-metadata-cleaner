from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class NCBIClient:
    base_url: str
    tool: str
    email: str
    api_key: Optional[str]
    timeout_s: float
    rps: float

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._last_t = 0.0

    def _throttle(self) -> None:
        if self.rps <= 0:
            return
        min_interval = 1.0 / self.rps
        now = time.time()
        dt = now - self._last_t
        if dt < min_interval:
            time.sleep(min_interval - dt)
        self._last_t = time.time()

    def get(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        self._throttle()
        p = dict(params)
        # NCBI recommends identifying tool + email; api_key optional.
        p["tool"] = self.tool
        p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = self._session.get(url, params=p, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp
