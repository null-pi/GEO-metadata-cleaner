import logging
import os
import pathlib

import typer
from rich.console import Console
from rich.table import Table

from .searcher import GEOSearcher
from .downloader import GEODownloader

geo_app = typer.Typer(help="Commands related to GEO dataset management")
console = Console()

logger = logging.getLogger(__name__)


@geo_app.command("search-and-download")
def search_and_download(
    query: str = typer.Argument(..., help="Search query for GEO datasets"),
    limit: int = typer.Option(
        5, "--limit", "-l", help="Maximum number of datasets to download"
    ),
    out_dir: str = typer.Option(
        None, "--out-dir", "-o", help="Output directory for downloaded datasets"
    ),
    save_list: pathlib.Path = typer.Option(
        None, "--save-list", "-s", help="File to save the list of found GSE IDs"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download of datasets even if they exist"
    ),
):
    searcher = GEOSearcher()
    gse_ids = searcher.search(query, retmax=limit)

    if not gse_ids:
        console.print("[bold red]No results found.[/bold red]")
        raise typer.Exit()

    table = Table(title=f"Found {len(gse_ids)} Studies")
    table.add_column("Accession", style="cyan")
    for gse in gse_ids:
        table.add_row(gse)
    console.print(table)

    if not typer.confirm(f"Download these {len(gse_ids)} datasets?"):
        raise typer.Exit()

    downloader = GEODownloader(out_dir)
    downloader.download(gse_ids, force=force)


@geo_app.command("download-list")
def download_list(
    file_path: pathlib.Path = typer.Argument(
        ..., help="Text file with one GSE ID per line"
    ),
    out_dir: pathlib.Path = typer.Option(None, "--out-dir", "-o", help="Output folder"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-download of datasets even if they exist"
    ),
):
    if not file_path.exists():
        console.print(f"[bold red]File {file_path} does not exist.[/bold red]")
        raise typer.Exit(code=1)

    with open(file_path, "r") as f:
        gse_ids = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    if not gse_ids:
        console.print("[bold red]No GSE IDs found in the file.[/bold red]")
        raise typer.Exit()

    downloader = GEODownloader(out_dir)
    downloader.download(gse_ids, force=force)
