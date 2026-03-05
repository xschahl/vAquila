"""Commandes cache Hugging Face."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from vaquila.cli_helpers import (
    cache_dir_to_model_id,
    dir_size_bytes,
    format_gb,
    list_cached_model_dirs,
    model_cache_repo_dir,
    purge_model_cache,
)
from vaquila.config import CONFIG
from vaquila.docker_service import list_managed_containers
from vaquila.exceptions import VaquilaError

console = Console()


def cmd_list_models() -> None:
    """Liste les modèles présents dans le cache Hugging Face local."""
    model_dirs = list_cached_model_dirs()
    if not model_dirs:
        console.print("[yellow]Aucun modèle trouvé dans le cache local.[/yellow]")
        return

    table = Table(title="vAquila - Modèles en cache")
    table.add_column("Modèle")
    table.add_column("Taille")
    table.add_column("Chemin cache")

    for model_dir in model_dirs:
        model_id = cache_dir_to_model_id(model_dir)
        size_bytes = dir_size_bytes(model_dir)
        table.add_row(model_id, format_gb(size_bytes), str(model_dir))

    console.print(table)


def cmd_rm_model(model_id: str) -> None:
    """Supprime un modèle du cache Hugging Face local."""
    try:
        managed_containers = list_managed_containers()
    except VaquilaError:
        managed_containers = []

    running_for_model = [
        container for container in managed_containers if container.model_id == model_id and container.status == "running"
    ]
    if running_for_model:
        names = ", ".join(container.name for container in running_for_model)
        console.print(
            f"[bold red]❌ Le modèle `{model_id}` est encore en cours d'exécution ({names}). "
            "Arrête-le d'abord avec `vaq stop`.[/bold red]"
        )
        raise typer.Exit(code=1)

    removed = purge_model_cache(model_id)
    if removed:
        console.print(
            f"[bold green]✅ Modèle supprimé du cache:[/bold green] "
            f"[cyan]{CONFIG.hf_cache_host_path / 'hub' / model_cache_repo_dir(model_id)}[/cyan]"
        )
        return

    console.print(f"[yellow]ℹ️ Modèle absent du cache: {model_id}[/yellow]")
