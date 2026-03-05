"""Commandes système/orchestration: ps, stop, rebalance, doctor, infer."""

from __future__ import annotations

import platform

import typer
from rich.console import Console
from rich.table import Table

from vaquila.cli_helpers import (
    LaunchPlan,
    check_hf_cache_path,
    estimate_shared_ratio_before_rebalance,
    format_gb,
    launch_plan_from_container,
    model_cache_repo_dir,
    purge_model_cache,
    rebalance_and_start,
)
from vaquila.config import CONFIG
from vaquila.docker_service import check_docker_connection, list_managed_containers, stop_model_container
from vaquila.exceptions import VaquilaError
from vaquila.gpu import read_all_gpu_snapshots, read_gpu_snapshot
from vaquila.inference import run_inference

console = Console()


def cmd_ps() -> None:
    """Liste les conteneurs vAquila actifs et leurs infos runtime."""
    try:
        snapshot_by_gpu = read_all_gpu_snapshots()
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
            format_gb(item.gpu_used_bytes),
        )

    console.print(table)


def cmd_stop(model_id: str, purge_cache: bool) -> None:
    """Stoppe et supprime le conteneur lié au modèle."""
    try:
        names = stop_model_container(model_id)
        joined = ", ".join(names)
        console.print(f"[bold green]✅ Conteneurs supprimés:[/bold green] [cyan]{joined}[/cyan]")
    except VaquilaError as exc:
        if purge_cache and "Aucun conteneur trouvé" in str(exc):
            console.print("[yellow]ℹ️ Aucun conteneur actif trouvé, purge du cache uniquement.[/yellow]")
        else:
            console.print(f"[bold red]❌ {exc}[/bold red]")
            raise typer.Exit(code=1)

    if purge_cache:
        removed = purge_model_cache(model_id)
        if removed:
            console.print(
                f"[bold green]✅ Cache modèle supprimé:[/bold green] "
                f"[cyan]{CONFIG.hf_cache_host_path / 'hub' / model_cache_repo_dir(model_id)}[/cyan]"
            )
        else:
            console.print("[yellow]ℹ️ Aucun cache local trouvé pour ce modèle.[/yellow]")


def cmd_rebalance(gpu_index: int, buffer_gb: float | None, startup_timeout: int, min_shared_ratio: float) -> None:
    """Rééquilibre les modèles déjà lancés sur un GPU."""
    try:
        auto_buffer = 2.0 if platform.system() == "Windows" else 1.5
        buffer = buffer_gb if buffer_gb is not None else auto_buffer

        running_on_same_gpu = [
            item for item in list_managed_containers() if item.gpu_index == gpu_index and item.status == "running"
        ]
        if len(running_on_same_gpu) < 2:
            raise VaquilaError("Le rééquilibrage manuel nécessite au moins 2 modèles running sur le même GPU.")

        plans: list[LaunchPlan] = [launch_plan_from_container(item) for item in running_on_same_gpu]

        current_snapshot = read_gpu_snapshot(gpu_index)
        estimated_ratio = estimate_shared_ratio_before_rebalance(
            snapshot=current_snapshot,
            buffer_gb=buffer,
            target_model_count=len(plans),
            running_models=running_on_same_gpu,
        )
        if estimated_ratio < min_shared_ratio:
            raise VaquilaError(
                "Rééquilibrage annulé avant relance vLLM: capacité VRAM insuffisante. "
                f"Ratio estimé={estimated_ratio:.3f}, seuil requis={min_shared_ratio:.3f}."
            )

        shared_ratio, started = rebalance_and_start(
            console=console,
            gpu_index=gpu_index,
            buffer_gb=buffer,
            plans=plans,
            min_shared_ratio=min_shared_ratio,
            startup_timeout=startup_timeout,
        )

        console.print("[bold green]✅ Rééquilibrage manuel terminé[/bold green]")
        for started_model, started_port, started_name in started:
            console.print(
                f"- [cyan]{started_model}[/cyan] | port [cyan]{started_port}[/cyan] | conteneur [cyan]{started_name}[/cyan]"
            )
        console.print(f"GPU: [cyan]{gpu_index}[/cyan] | Ratio partagé: [cyan]{shared_ratio:.3f}[/cyan]")
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)


def cmd_doctor(gpu_index: int) -> None:
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
        cache_path = check_hf_cache_path()
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


def cmd_infer(
    model_id: str,
    prompt: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> None:
    """Teste l'inférence d'un modèle déjà lancé via l'API vLLM."""
    try:
        answer = run_inference(
            base_url=base_url,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout,
        )
        console.print("[bold green]✅ Réponse modèle[/bold green]")
        console.print(answer)
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)
