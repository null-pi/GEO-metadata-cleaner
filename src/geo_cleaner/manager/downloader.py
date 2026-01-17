import logging
import os
import pathlib

from rich.console import Console

import requests
from tqdm import tqdm

from geo_cleaner.database import GEODatabase, DownloadRecord

console = Console()
logger = logging.getLogger(__name__)


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series"


class GEODownloader:
    def __init__(self, out_dir: str):
        current_script_dir = pathlib.Path(__file__).resolve()
        src_root = current_script_dir.parent.parent.parent

        if out_dir is None:
            self.out_dir = src_root / "resources" / "geo"
        else:
            self.out_dir = pathlib.Path(out_dir)

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.db = GEODatabase()

    def _construct_ftp_url(self, gse_ids: str) -> str:
        clean_id = gse_ids.strip().upper()

        if len(clean_id) < 6:
            stub = "GSEnnn"
        else:
            stub = clean_id[:-3] + "nnn"

        filename = f"{clean_id}_family.xml.tgz"

        return f"{BASE_URL}/{stub}/{clean_id}/miniml/{filename}"

    def download(
        self, gse_ids: list[str], force: bool = False
    ) -> list[tuple[str, pathlib.Path]]:
        console.print(f"⬇️  Queueing {len(gse_ids)} datasets for download...")

        successful_downloads = []

        for gse in tqdm(gse_ids, desc="Total Progress"):
            url = self._construct_ftp_url(gse)
            filename = self.out_dir / f"{gse}_family.xml.tgz"
            temp_filename = self.out_dir / f"{gse}_family.xml.tgz.tmp"

            if filename.exists() and not force:
                tqdm.write(f"✅ {gse} exists. Skipping.")
                continue

            try:
                with requests.get(url, stream=True) as r:
                    if r.status_code == 404:
                        tqdm.write(f"❌ {gse} not found (404). Check ID.")
                        continue
                    r.raise_for_status()

                    total = int(r.headers.get("content-length", 0))

                    with (
                        open(temp_filename, "wb") as f,
                        tqdm(
                            desc=gse,
                            total=total,
                            unit="iB",
                            unit_scale=True,
                            leave=False,
                        ) as bar,
                    ):
                        for chunk in r.iter_content(chunk_size=1024):
                            f.write(chunk)
                            bar.update(len(chunk))

                os.rename(temp_filename, filename)
                tqdm.write(f"✅ Downloaded {gse} successfully.")

                record = DownloadRecord(gse_id=gse, filename=filename, query=url)
                
                self.db.add_record(record)

            except Exception as e:
                tqdm.write(f"❌ Error downloading {gse}: {e}")
                if temp_filename.exists():
                    os.remove(temp_filename)

        return successful_downloads
