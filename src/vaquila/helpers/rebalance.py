"""Helpers de rééquilibrage multi-modèles."""

from __future__ import annotations

from rich.console import Console

from vaquila.config import CONFIG
from vaquila.docker_service import run_model_container, stop_containers_by_name
from vaquila.exceptions import VaquilaError
from vaquila.gpu import read_gpu_snapshot
from vaquila.helpers.runtime import estimate_required_ratio, normalize_optional_text
from vaquila.helpers.startup import wait_until_model_ready
from vaquila.helpers.types import LaunchPlan
from vaquila.models import GpuSnapshot, ManagedContainer


def launch_plan_from_container(container: ManagedContainer) -> LaunchPlan:
    """Construit un plan de relance depuis un conteneur déjà géré par vAquila."""
    if container.host_port is None:
        raise VaquilaError(f"Port introuvable pour {container.name}, impossible de rééquilibrer.")

    max_num_seqs = container.max_num_seqs or 1
    max_model_len = container.max_model_len or 16384
    tool_call_parser = normalize_optional_text(container.tool_call_parser)
    reasoning_parser = normalize_optional_text(container.reasoning_parser)
    enable_thinking = container.enable_thinking if container.enable_thinking is not None else True
    required_ratio = container.required_ratio
    if required_ratio is None:
        required_ratio = estimate_required_ratio(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            enable_thinking=enable_thinking,
        )

    allow_long_context_override = bool(container.allow_long_context_override)

    return LaunchPlan(
        model_id=container.model_id,
        host_port=container.host_port,
        existing_name=container.name,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        enable_thinking=enable_thinking,
        required_ratio=required_ratio,
        allow_long_context_override=allow_long_context_override,
    )


def compute_shared_ratio(snapshot: GpuSnapshot, buffer_gb: float, model_count: int) -> tuple[float, float]:
    """Calcule un ratio partagé pour lancer plusieurs modèles sur le même GPU."""
    if model_count <= 0:
        raise VaquilaError("Le nombre de modèles à répartir doit être >= 1.")

    buffer_bytes = int(buffer_gb * (1024**3))
    usable_bytes = snapshot.free_bytes - buffer_bytes
    if usable_bytes <= 0:
        free_gb = snapshot.free_bytes / (1024**3)
        raise VaquilaError(
            f"VRAM insuffisante pour un lancement partagé: {free_gb:.2f} Gio libres, "
            f"buffer demandé {buffer_gb:.2f} Gio."
        )

    shared_raw = (usable_bytes / snapshot.total_bytes) / model_count
    minimum_ratio = 0.03
    if shared_raw < minimum_ratio:
        raise VaquilaError(
            f"VRAM insuffisante pour {model_count} modèles simultanés sur ce GPU "
            f"(ratio partagé estimé={shared_raw:.3f})."
        )

    return min(shared_raw, 0.98), buffer_gb


def estimate_shared_ratio_before_rebalance(
    snapshot: GpuSnapshot,
    buffer_gb: float,
    target_model_count: int,
    running_models: list[ManagedContainer],
) -> float:
    """Estime un ratio partagé réaliste avant de relancer des modèles."""
    if target_model_count <= 0:
        raise VaquilaError("Le nombre de modèles à répartir doit être >= 1.")

    buffer_ratio = (buffer_gb * (1024**3)) / snapshot.total_bytes
    used_ratio = snapshot.used_bytes / snapshot.total_bytes
    reserved_by_vaquila = sum(item.gpu_utilization or 0.0 for item in running_models)
    non_vaquila_ratio = max(0.0, used_ratio - reserved_by_vaquila)

    available_ratio_after_restart = 1.0 - buffer_ratio - non_vaquila_ratio
    if available_ratio_after_restart <= 0:
        return 0.0

    return min(0.98, available_ratio_after_restart / target_model_count)


def rebalance_and_start(
    console: Console,
    gpu_index: int,
    buffer_gb: float,
    plans: list[LaunchPlan],
    min_shared_ratio: float,
    startup_timeout: int,
) -> tuple[float, list[tuple[str, int, str]]]:
    """Rééquilibre la VRAM entre plusieurs modèles puis les relance."""
    if not plans:
        raise VaquilaError("Aucun modèle à lancer pour le rééquilibrage.")

    ports = [plan.host_port for plan in plans]
    if len(set(ports)) != len(ports):
        raise VaquilaError("Conflit de ports détecté pendant le rééquilibrage.")

    existing_names = [plan.existing_name for plan in plans if plan.existing_name]
    removed = stop_containers_by_name(existing_names)
    if removed:
        console.print(
            "[yellow]⚠️ Rééquilibrage GPU: conteneurs stoppés temporairement: "
            f"{', '.join(removed)}[/yellow]"
        )

    fresh_snapshot = read_gpu_snapshot(gpu_index)
    max_available_shared_ratio, _ = compute_shared_ratio(fresh_snapshot, buffer_gb, len(plans))
    max_required_ratio = max(plan.required_ratio for plan in plans)
    target_min_ratio = max(min_shared_ratio, max_required_ratio)
    if max_available_shared_ratio < target_min_ratio:
        raise VaquilaError(
            "Rééquilibrage refusé: capacité VRAM insuffisante pour la configuration demandée. "
            f"Ratio max dispo={max_available_shared_ratio:.3f}, ratio requis={target_min_ratio:.3f}."
        )

    shared_ratio = target_min_ratio
    console.print(
        "[cyan]Répartition VRAM multi-modèles activée: "
        f"{len(plans)} modèle(s), ratio appliqué={shared_ratio:.3f} (max dispo={max_available_shared_ratio:.3f})[/cyan]"
    )

    started: list[tuple[str, int, str]] = []
    for plan in plans:
        with console.status(
            f"[cyan]Lancement {plan.model_id} sur le port {plan.host_port} (ratio {shared_ratio:.3f})...[/cyan]",
            spinner="dots",
        ):
            container = run_model_container(
                model_id=plan.model_id,
                host_port=plan.host_port,
                gpu_index=gpu_index,
                gpu_utilization=shared_ratio,
                max_num_seqs=plan.max_num_seqs,
                max_model_len=plan.max_model_len,
                tool_call_parser=plan.tool_call_parser,
                reasoning_parser=plan.reasoning_parser,
                enable_thinking=plan.enable_thinking,
                required_ratio=plan.required_ratio,
                allow_long_context_override=plan.allow_long_context_override,
                config=CONFIG,
            )

        wait_until_model_ready(console, container.name, timeout_seconds=startup_timeout)
        started.append((plan.model_id, plan.host_port, container.name))

    return shared_ratio, started
