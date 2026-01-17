from dataclasses import dataclass
import logging
import os
import pathlib
import json
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class OntologyConfig:
    name: str
    url: str
    desc: str
    filename: str


class OntologyBuilder:
    def __init__(
        self, config_file: Optional[str] = None, out_dir: Optional[str] = None
    ):
        current_script_dir = pathlib.Path(__file__).resolve()
        src_root = current_script_dir.parent.parent.parent

        if config_file is None:
            self.config_file = src_root / "ontology.json"
        else:
            self.config_file = pathlib.Path(config_file)

        if out_dir is None:
            self.out_dir = src_root / "resources" / "ontologies"
        else:
            self.out_dir = pathlib.Path(out_dir)

        logger.info(f"Looking for ontologies config at: {self.config_file}")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._build_config()

    def _build_config(self):
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file {self.config_file} not found.")

        with open(self.config_file, "r") as f:
            config: dict[str, dict[str, str]] = json.load(f)

        self.ontologies: list[OntologyConfig] = []
        for ontology_name, ontology_values in config.items():
            filepath = pathlib.Path(self.out_dir) / f"{ontology_name}.obo"

            self.ontologies.append(
                OntologyConfig(
                    name=ontology_name,
                    url=ontology_values["url"],
                    desc=ontology_values.get("desc", ""),
                    filename=str(filepath),
                )
            )

        logger.info(f"Prepared {len(self.ontologies)} ontologies for download.")

    def download(self, force: bool = False):
        for ontology in tqdm(self.ontologies, desc="Overall Progress"):
            try:
                temp_filename = ontology.filename + ".tmp"
                final_filename = ontology.filename

                if os.path.exists(final_filename) and not force:
                    tqdm.write(
                        f"✅ {ontology.name} exists and matches size. Skipping."
                    )
                    continue
                        

                with requests.get(ontology.url, stream=True) as response:
                    response.raise_for_status()
                    total_download_size = int(response.headers.get("content-length", 0))

                    with (
                        open(temp_filename, "wb") as f,
                        tqdm(
                            desc=f"Downloading {ontology.name}",
                            total=(
                                total_download_size if total_download_size > 0 else None
                            ),
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                            leave=False,
                        ) as inner_bar,
                    ):
                        for chunk in response.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            inner_bar.update(size)

                os.replace(temp_filename, final_filename)
                tqdm.write(f"✨ Saved {ontology.name}")

            except Exception as e:
                tqdm.write(f"❌ Failed to download {ontology.name}: {e}")
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)


if __name__ == "__main__":
    builder = OntologyBuilder()
    if hasattr(builder, "ontologies"):
        builder.download()
