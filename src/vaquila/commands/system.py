"""System/orchestration commands: ps, stop, doctor, infer."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from vaquila.cli_helpers import (
    check_hf_cache_path,
    format_gb,
    model_cache_repo_dir,
    purge_model_cache,
)
from vaquila.config import CONFIG
from vaquila.docker_service import check_docker_connection, list_managed_containers, stop_model_container
from vaquila.exceptions import VaquilaError
from vaquila.gpu import read_all_gpu_snapshots, read_gpu_snapshot
from vaquila.inference import run_inference

console = Console()


def cmd_ps() -> None:
    """List active vAquila containers and runtime details."""
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
        console.print("[yellow]No vAquila containers found.[/yellow]")
        return

    table = Table(title="vAquila - Containers")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Port")
    table.add_column("Backend")
    table.add_column("GPU")
    table.add_column("VRAM Used")

    for item in containers:
        table.add_row(
            item.name,
            item.model_id,
            item.status,
            str(item.host_port) if item.host_port is not None else "n/a",
            (item.compute_backend or "gpu").upper(),
            str(item.gpu_index) if item.gpu_index is not None else "n/a",
            format_gb(item.gpu_used_bytes),
        )

    console.print(table)


def cmd_stop(model_id: str, purge_cache: bool) -> None:
    """Stop and remove the container linked to a model."""
    try:
        names = stop_model_container(model_id)
        joined = ", ".join(names)
        console.print(f"[bold green]✅ Containers removed:[/bold green] [cyan]{joined}[/cyan]")
    except VaquilaError as exc:
        if purge_cache and "No container found" in str(exc):
            console.print("[yellow]ℹ️ No running container found, cache purge only.[/yellow]")
        else:
            console.print(f"[bold red]❌ {exc}[/bold red]")
            raise typer.Exit(code=1)

    if purge_cache:
        removed = purge_model_cache(model_id)
        if removed:
            console.print(
                f"[bold green]✅ Model cache removed:[/bold green] "
                f"[cyan]{CONFIG.hf_cache_host_path / 'hub' / model_cache_repo_dir(model_id)}[/cyan]"
            )
        else:
            console.print("[yellow]ℹ️ No local cache found for this model.[/yellow]")


def cmd_doctor(gpu_index: int) -> None:
    """Validate runtime environment (Docker, GPU, cache)."""
    checks: list[tuple[str, bool, str]] = []

    try:
        check_docker_connection()
        checks.append(("Docker daemon", True, "Connection OK"))
    except VaquilaError as exc:
        checks.append(("Docker daemon", False, str(exc)))

    try:
        snapshot = read_gpu_snapshot(gpu_index)
        details = (
            f"GPU {gpu_index} detected | total={snapshot.total_bytes / (1024**3):.2f} GiB "
            f"free={snapshot.free_bytes / (1024**3):.2f} GiB"
        )
        checks.append(("NVIDIA / NVML", True, details))
    except VaquilaError as exc:
        checks.append(("NVIDIA / NVML", False, str(exc)))

    try:
        cache_path = check_hf_cache_path()
        checks.append(("Hugging Face cache", True, cache_path))
    except VaquilaError as exc:
        checks.append(("Hugging Face cache", False, str(exc)))

    table = Table(title="vAquila Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")

    has_failure = False
    for label, ok, details in checks:
        status = "[green]OK[/green]" if ok else "[red]FAILED[/red]"
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
    """Test inference against an already running model via the vLLM API."""
    try:
        answer = run_inference(
            base_url=base_url,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout,
        )
        console.print("[bold green]✅ Model response[/bold green]")
        console.print(answer)
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)
