"""Entrée CLI principale de vAquila."""

from __future__ import annotations

import os
import typer
from rich.console import Console
from rich.table import Table

from vaquila.config import CONFIG
from vaquila.docker_service import (
    check_docker_connection,
    list_managed_containers,
    run_model_container,
    stop_model_container,
)
from vaquila.exceptions import VaquilaError
from vaquila.gpu import compute_gpu_memory_utilization, read_gpu_snapshot

app = typer.Typer(help="vAquila - Orchestration vLLM + Docker")
console = Console()


def _format_gb(value_bytes: int | None) -> str:
    """Affiche des octets en Gio lisibles."""
    if value_bytes is None:
        return "n/a"
    return f"{value_bytes / (1024**3):.2f} Gio"


def _check_hf_cache_path() -> str:
    """Vérifie que le cache Hugging Face est accessible en lecture/écriture."""
    path = CONFIG.hf_cache_host_path
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise VaquilaError(f"Le chemin de cache n'est pas un dossier: {path}")

    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    if not readable or not writable:
        raise VaquilaError(f"Permissions insuffisantes sur le cache: {path} (read={readable}, write={writable})")

    return str(path)


@app.command("run")
def run(
    model_id: str = typer.Argument(..., help="Model id Hugging Face, ex: meta-llama/Llama-3-8B-Instruct"),
    port: int = typer.Option(CONFIG.default_host_port, "--port", "-p", help="Port hôte exposé"),
    gpu_index: int = typer.Option(0, "--gpu", help="Index GPU NVIDIA"),
    buffer_gb: float = typer.Option(
        CONFIG.security_vram_buffer_gb,
        "--buffer-gb",
        help="Buffer VRAM en Gio réservé pour l'OS/processus tiers",
    ),
) -> None:
    """Lance un modèle dans un conteneur vLLM en arrière-plan."""
    try:
        snapshot = read_gpu_snapshot(gpu_index)
        ratio = compute_gpu_memory_utilization(snapshot, security_buffer_gb=buffer_gb)

        container = run_model_container(
            model_id=model_id,
            host_port=port,
            gpu_index=gpu_index,
            gpu_utilization=ratio,
            config=CONFIG,
        )

        console.print("[bold green]✅ Modèle lancé[/bold green]")
        console.print(f"Conteneur: [cyan]{container.name}[/cyan]")
        console.print(f"API vLLM: [cyan]http://localhost:{port}[/cyan]")
        console.print(f"GPU: [cyan]{gpu_index}[/cyan] | Utilization: [cyan]{ratio:.3f}[/cyan]")
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)


@app.command("ps")
def ps() -> None:
    """Liste les conteneurs vAquila actifs et leurs infos runtime."""
    try:
        snapshot = read_gpu_snapshot(0)
        snapshot_by_gpu = {0: snapshot}
    except VaquilaError:
        snapshot_by_gpu = None

    try:
        containers = list_managed_containers(snapshot_by_gpu=snapshot_by_gpu)
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)

    if not containers:
        console.print("[yellow]Aucun conteneur vAquila trouvé.[/yellow]")
        return

    table = Table(title="vAquila - Conteneurs")
    table.add_column("Nom")
    table.add_column("Modèle")
    table.add_column("Statut")
    table.add_column("Port")
    table.add_column("GPU")
    table.add_column("VRAM utilisée")

    for item in containers:
        table.add_row(
            item.name,
            item.model_id,
            item.status,
            str(item.host_port) if item.host_port is not None else "n/a",
            str(item.gpu_index) if item.gpu_index is not None else "n/a",
            _format_gb(item.gpu_used_bytes),
        )

    console.print(table)


@app.command("stop")
def stop(
    model_id: str = typer.Argument(..., help="Model id Hugging Face à arrêter"),
) -> None:
    """Stoppe et supprime le conteneur lié au modèle."""
    try:
        name = stop_model_container(model_id)
        console.print(f"[bold green]✅ Conteneur supprimé:[/bold green] [cyan]{name}[/cyan]")
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)


@app.command("doctor")
def doctor(
    gpu_index: int = typer.Option(0, "--gpu", help="Index GPU NVIDIA à vérifier"),
) -> None:
    """Vérifie l'environnement d'exécution (Docker, GPU, cache)."""
    checks: list[tuple[str, bool, str]] = []

    try:
        check_docker_connection()
        checks.append(("Docker daemon", True, "Connexion OK"))
    except VaquilaError as exc:
        checks.append(("Docker daemon", False, str(exc)))

    try:
        snapshot = read_gpu_snapshot(gpu_index)
        details = (
            f"GPU {gpu_index} détecté | total={snapshot.total_bytes / (1024**3):.2f} Gio "
            f"free={snapshot.free_bytes / (1024**3):.2f} Gio"
        )
        checks.append(("NVIDIA / NVML", True, details))
    except VaquilaError as exc:
        checks.append(("NVIDIA / NVML", False, str(exc)))

    try:
        cache_path = _check_hf_cache_path()
        checks.append(("Cache Hugging Face", True, cache_path))
    except VaquilaError as exc:
        checks.append(("Cache Hugging Face", False, str(exc)))

    table = Table(title="vAquila Doctor")
    table.add_column("Check")
    table.add_column("Statut")
    table.add_column("Détails")

    has_failure = False
    for label, ok, details in checks:
        status = "[green]OK[/green]" if ok else "[red]KO[/red]"
        if not ok:
            has_failure = True
        table.add_row(label, status, details)

    console.print(table)

    if has_failure:
        raise typer.Exit(code=1)
