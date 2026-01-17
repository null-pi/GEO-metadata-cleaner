import logging
import os
import pathlib

import typer
from rich.console import Console
from rich.table import Table

from geo_cleaner.logging_conf import setup_logging
from geo_cleaner.ontology.builder import OntologyBuilder
from geo_cleaner.utils import get_size_str


setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(help="GEO Metadata Cleaner CLI")
console = Console()

ontology_app = typer.Typer(help="Commands related to ontology management")
app.add_typer(ontology_app, name="ontology")


@ontology_app.command("download")
def download_ontology(
    config: pathlib.Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the configuration file",
    ),
    out_dir: pathlib.Path = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Output directory to save the downloaded ontologies",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download of ontologies even if they already exist",
    ),
):
    console.print(f"[bold blue]🚀 Starting Ontology Download[/bold blue]")
    console.print(f"Reading from: [green]{config}[/green]")
    console.print(f"Saving to:    [green]{out_dir}[/green]")

    if force:
        console.print("[yellow]⚠️  Force Mode: Overwriting existing files[/yellow]")

    try:
        builder = OntologyBuilder(
            config_file=config,
            out_dir=out_dir,
        )

        builder.download(force=force)

        console.print(
            "\n[bold green]✅ All downloads finished successfully![/bold green]"
        )

    except Exception as e:
        logger.exception("Download process failed")
        console.print(f"[bold red]Critical Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@ontology_app.command("list")
def list_ontologies(
    config: pathlib.Path = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    out_dir: pathlib.Path = typer.Option(
        None, "--out-dir", "-o", help="Output directory to check"
    ),
):
    """
    List configured ontologies and their download status/size.
    """
    try:
        # We initialize the builder just to load the config and resolve paths
        builder = OntologyBuilder(config_file=config, out_dir=out_dir)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    table = Table(title="Ontology Status")

    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Local Path", style="dim")

    for ont in builder.ontologies:
        file_path = pathlib.Path(ont.filename)

        if file_path.exists():
            status = "[bold green]Downloaded[/bold green]"
            size = get_size_str(file_path)
        else:
            status = "[bold red]Missing[/bold red]"
            size = "-"

        try:
            display_path = file_path.relative_to(os.getcwd())
        except ValueError:
            display_path = file_path

        table.add_row(ont.name, ont.desc, status, size, str(display_path))

    console.print(table)
