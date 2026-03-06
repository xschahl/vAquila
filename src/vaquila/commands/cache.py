"""Hugging Face cache commands."""

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
    """List models available in the local Hugging Face cache."""
    model_dirs = list_cached_model_dirs()
    if not model_dirs:
        console.print("[yellow]No model found in local cache.[/yellow]")
        return

    table = Table(title="vAquila - Cached Models")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Cache Path")

    for model_dir in model_dirs:
        model_id = cache_dir_to_model_id(model_dir)
        size_bytes = dir_size_bytes(model_dir)
        table.add_row(model_id, format_gb(size_bytes), str(model_dir))

    console.print(table)


def cmd_rm_model(model_id: str) -> None:
    """Remove a model from the local Hugging Face cache."""
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
            f"[bold red]❌ Model `{model_id}` is still running ({names}). "
            "Stop it first with `vaq stop`.[/bold red]"
        )
        raise typer.Exit(code=1)

    removed = purge_model_cache(model_id)
    if removed:
        console.print(
            f"[bold green]✅ Model removed from cache:[/bold green] "
            f"[cyan]{CONFIG.hf_cache_host_path / 'hub' / model_cache_repo_dir(model_id)}[/cyan]"
        )
        return

    console.print(f"[yellow]ℹ️ Model not found in cache: {model_id}[/yellow]")
