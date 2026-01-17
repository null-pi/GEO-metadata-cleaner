import logging
import os
from typing import Any

import requests

from rich.console import Console

console = Console()

logger = logging.getLogger(__name__)

NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class GEOSearcher:
    def __init__(self, email: str = NCBI_EMAIL, api_key: str = NCBI_API_KEY):
        if not email or not api_key:
            raise ValueError(
                "NCBI_EMAIL and NCBI_API_KEY must be set in environment variables."
            )

        self.params: dict[str, str] = {
            "db": "gds",
            "retmode": "json",
            "email": email,
            "api_key": api_key,
        }

    def search(self, term: str, retmax: int = 20) -> list[str]:
        final_query = f'{term} AND "gse"[Entry Type]'

        console.print(f"🔍 Searching GEO for term: [cyan]'{final_query}'[/cyan]")

        payload = self.params.copy()
        payload.update({"term": final_query, "retmax": str(retmax)})

        try:
            response = requests.get(BASE_URL, params=payload)
            response.raise_for_status()

            data: dict[str, Any] = response.json()

            uids: list[str] = data.get("esearchresult", {}).get("idlist", [])

            return self._uids_to_accessions(uids)

        except Exception as e:
            logger.exception("GEO search failed")
            console.print(f"[bold red]Error during GEO search:[/bold red] {e}")
            raise e

    def _uids_to_accessions(self, uids: list[str]) -> list[str]:
        if not uids:
            return []

        payload = self.params.copy()
        payload.update({"id": ",".join(uids)})

        try:
            response = requests.get(SUMMARY_URL, params=payload)
            response.raise_for_status()

            data: dict[str, Any] = response.json()

            gse_ids: list[str] = []
            results = data.get("result", {})
            for uid in uids:
                if uid in results:
                    acc: str = results[uid].get("accession", "")
                    if acc.startswith("GSE"):
                        gse_ids.append(acc)

            return gse_ids
        except Exception as e:
            logger.exception("Failed to convert UIDs to accessions")
            console.print(
                f"[bold red]Error during UID to accession conversion:[/bold red] {e}"
            )
            raise e
