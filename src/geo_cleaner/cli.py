import logging
from dotenv import load_dotenv

load_dotenv()

import typer

from geo_cleaner.logging_conf import setup_logging
from geo_cleaner.ontology.cli import ontology_app
from geo_cleaner.manager.cli import geo_app

setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(help="GEO Metadata Cleaner CLI")

app.add_typer(ontology_app, name="ontology")
app.add_typer(geo_app, name="geo")
