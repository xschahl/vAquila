"""Commande de lancement de modèle (run)."""

from __future__ import annotations

from contextlib import suppress
import platform

import typer
from rich.console import Console

from vaquila.cli_helpers import (
    LaunchPlan,
    estimate_required_ratio,
    estimate_shared_ratio_before_rebalance,
    extract_kv_max_concurrency,
    is_retryable_vram_error,
    launch_plan_from_container,
    ratio_candidates,
    rebalance_and_start,
    resolve_context_strategy,
    resolve_run_runtime_settings,
    wait_until_model_ready,
)
from vaquila.config import CONFIG
from vaquila.docker_service import (
    get_container,
    list_managed_containers,
    run_model_container,
    stop_containers_by_name,
)
from vaquila.exceptions import VaquilaError
from vaquila.gpu import compute_adaptive_gpu_memory_utilization, compute_gpu_memory_utilization, read_gpu_snapshot

console = Console()


def cmd_run(
    model_id: str,
    port: int,
    gpu_index: int,
    buffer_gb: float | None,
    startup_timeout: int,
    rebalance_existing: bool,
    min_shared_ratio: float,
    max_num_seqs: int | None,
    max_model_len: int | None,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool | None,
    allow_long_context_override: bool | None,
    share_gpu: bool,
) -> None:
    """Lance un modèle dans un conteneur vLLM en arrière-plan."""
    try:
        auto_buffer = 2.0 if platform.system() == "Windows" else 1.5
        buffer = buffer_gb if buffer_gb is not None else auto_buffer
        (
            resolved_max_num_seqs,
            resolved_max_model_len,
            resolved_tool_call_parser,
            resolved_reasoning_parser,
            resolved_enable_thinking,
        ) = resolve_run_runtime_settings(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            enable_thinking=enable_thinking,
        )

        resolved_max_model_len, resolved_allow_long_context_override = resolve_context_strategy(
            console=console,
            model_id=model_id,
            requested_max_model_len=resolved_max_model_len,
            allow_long_context_override=allow_long_context_override,
        )

        new_required_ratio = estimate_required_ratio(
            max_num_seqs=resolved_max_num_seqs,
            max_model_len=resolved_max_model_len,
            tool_call_parser=resolved_tool_call_parser,
            reasoning_parser=resolved_reasoning_parser,
            enable_thinking=resolved_enable_thinking,
        )

        snapshot = read_gpu_snapshot(gpu_index)
        running_on_same_gpu = [
            item for item in list_managed_containers() if item.gpu_index == gpu_index and item.status == "running"
        ]

        if running_on_same_gpu and rebalance_existing:
            existing_plans: list[LaunchPlan] = [launch_plan_from_container(item) for item in running_on_same_gpu]

            if not share_gpu:
                consent = typer.confirm(
                    (
                        f"{len(existing_plans)} modèle(s) tourne(nt) déjà sur le GPU {gpu_index}. "
                        "Activer le mode partagé et rééquilibrer la VRAM ?"
                    ),
                    default=False,
                )
                if not consent:
                    raise VaquilaError("Lancement annulé. Relance avec `--share-gpu` pour autoriser le rééquilibrage.")

            target_model_count = len(existing_plans) + 1
            estimated_ratio = estimate_shared_ratio_before_rebalance(
                snapshot=snapshot,
                buffer_gb=buffer,
                target_model_count=target_model_count,
                running_models=running_on_same_gpu,
            )
            target_required_ratio = max(
                [new_required_ratio, min_shared_ratio] + [plan.required_ratio for plan in existing_plans]
            )
            if estimated_ratio < target_required_ratio:
                raise VaquilaError(
                    "Lancement annulé avant démarrage vLLM: capacité VRAM insuffisante pour le partage. "
                    f"Ratio estimé={estimated_ratio:.3f}, seuil requis={target_required_ratio:.3f}."
                )

            plans = [
                LaunchPlan(
                    model_id=model_id,
                    host_port=port,
                    existing_name=None,
                    max_num_seqs=resolved_max_num_seqs,
                    max_model_len=resolved_max_model_len,
                    tool_call_parser=resolved_tool_call_parser,
                    reasoning_parser=resolved_reasoning_parser,
                    enable_thinking=resolved_enable_thinking,
                    required_ratio=new_required_ratio,
                    allow_long_context_override=resolved_allow_long_context_override,
                )
            ] + existing_plans
            shared_ratio, started = rebalance_and_start(
                console=console,
                gpu_index=gpu_index,
                buffer_gb=buffer,
                plans=plans,
                min_shared_ratio=min_shared_ratio,
                startup_timeout=startup_timeout,
            )

            console.print("[bold green]✅ Rééquilibrage terminé[/bold green]")
            for started_model, started_port, started_name in started:
                console.print(
                    f"- [cyan]{started_model}[/cyan] | port [cyan]{started_port}[/cyan] | conteneur [cyan]{started_name}[/cyan]"
                )
            console.print(f"GPU: [cyan]{gpu_index}[/cyan] | Ratio partagé: [cyan]{shared_ratio:.3f}[/cyan]")
            return

        try:
            max_available_ratio = compute_gpu_memory_utilization(snapshot, security_buffer_gb=buffer)
        except VaquilaError:
            if not running_on_same_gpu:
                raise
            max_available_ratio, adaptive_buffer = compute_adaptive_gpu_memory_utilization(
                snapshot,
                security_buffer_gb=buffer,
            )
            console.print(
                "[yellow]⚠️ VRAM fragmentée par des modèles déjà actifs: "
                f"buffer ajusté à {adaptive_buffer:.2f} Gio, ratio max dispo={max_available_ratio:.3f}.[/yellow]"
            )

        if max_available_ratio < new_required_ratio:
            raise VaquilaError(
                "Configuration runtime trop exigeante pour la VRAM disponible avant lancement vLLM. "
                f"Ratio max dispo={max_available_ratio:.3f}, ratio requis={new_required_ratio:.3f}. "
                "Réduis max-num-seqs, max-model-len, ou désactive certains parsers/thinking."
            )

        ratio = max(new_required_ratio, 0.02)

        typer.secho(
            f"[vAquila] Buffer VRAM utilisé: {buffer:.2f} Gio | Ratio appliqué (minimal): {ratio:.3f} | Ratio max dispo: {max_available_ratio:.3f}",
            fg=typer.colors.CYAN,
        )
        console.print(
            "[cyan]Runtime demandé:[/cyan] "
            f"max_num_seqs={resolved_max_num_seqs}, max_model_len={resolved_max_model_len}, "
            f"tool_call_parser={resolved_tool_call_parser or 'none'}, "
            f"reasoning_parser={resolved_reasoning_parser or 'none'}, "
            f"enable_thinking={resolved_enable_thinking}, ratio requis~{new_required_ratio:.3f}"
        )

        container = None
        selected_ratio = ratio
        attempt_ratios = ratio_candidates(min_ratio=ratio, max_ratio=max_available_ratio)
        if not attempt_ratios:
            raise VaquilaError(
                "Impossible de calculer un ratio de lancement valide. "
                f"min={ratio:.3f}, max={max_available_ratio:.3f}"
            )

        for attempt_index, attempt_ratio in enumerate(attempt_ratios, start=1):
            selected_ratio = attempt_ratio
            with console.status(
                "[cyan]Création du conteneur vLLM et préparation du téléchargement Hugging Face...[/cyan]",
                spinner="dots",
            ):
                container = run_model_container(
                    model_id=model_id,
                    host_port=port,
                    gpu_index=gpu_index,
                    gpu_utilization=attempt_ratio,
                    max_num_seqs=resolved_max_num_seqs,
                    max_model_len=resolved_max_model_len,
                    tool_call_parser=resolved_tool_call_parser,
                    reasoning_parser=resolved_reasoning_parser,
                    enable_thinking=resolved_enable_thinking,
                    required_ratio=new_required_ratio,
                    allow_long_context_override=resolved_allow_long_context_override,
                    config=CONFIG,
                )

            try:
                wait_until_model_ready(console, container.name, timeout_seconds=startup_timeout)
                break
            except VaquilaError as exc:
                with suppress(VaquilaError):
                    stop_containers_by_name([container.name])

                if attempt_index == len(attempt_ratios) or not is_retryable_vram_error(str(exc)):
                    raise

                next_ratio = attempt_ratios[attempt_index]
                console.print(
                    "[yellow]⚠️ Ratio insuffisant pour le KV cache, tentative suivante: "
                    f"{next_ratio:.3f} (précédent={attempt_ratio:.3f})[/yellow]"
                )

        runtime_container = get_container(container.name)
        runtime_logs = runtime_container.logs(tail=400).decode("utf-8", errors="replace")
        observed_concurrency = extract_kv_max_concurrency(runtime_logs, resolved_max_model_len)
        target_concurrency = max(1.0, resolved_max_num_seqs * 1.5)
        upper_concurrency_bound = target_concurrency * 1.25

        if observed_concurrency is not None and observed_concurrency > upper_concurrency_bound:
            previous_ratio = selected_ratio
            downscale_ratio = round(
                max(
                    new_required_ratio,
                    0.02,
                    selected_ratio * (target_concurrency / observed_concurrency) * 1.15,
                    selected_ratio * 0.80,
                ),
                3,
            )
            downscale_ratio = min(downscale_ratio, round(previous_ratio - 0.005, 3))

            if downscale_ratio < previous_ratio:
                console.print(
                    "[yellow]⚠️ Sur-provisionnement KV détecté: "
                    f"concurrence observée={observed_concurrency:.2f}x, cible~{target_concurrency:.2f}x. "
                    f"Tentative d'optimisation du ratio: {previous_ratio:.3f} -> {downscale_ratio:.3f}[/yellow]"
                )

                with suppress(VaquilaError):
                    stop_containers_by_name([container.name])

                tuned_container = None
                try:
                    with console.status("[cyan]Relance optimisée avec ratio réduit...[/cyan]", spinner="dots"):
                        tuned_container = run_model_container(
                            model_id=model_id,
                            host_port=port,
                            gpu_index=gpu_index,
                            gpu_utilization=downscale_ratio,
                            max_num_seqs=resolved_max_num_seqs,
                            max_model_len=resolved_max_model_len,
                            tool_call_parser=resolved_tool_call_parser,
                            reasoning_parser=resolved_reasoning_parser,
                            enable_thinking=resolved_enable_thinking,
                            required_ratio=new_required_ratio,
                            allow_long_context_override=resolved_allow_long_context_override,
                            config=CONFIG,
                        )

                    wait_until_model_ready(console, tuned_container.name, timeout_seconds=startup_timeout)
                    container = tuned_container
                    selected_ratio = downscale_ratio
                except VaquilaError as tune_exc:
                    with suppress(VaquilaError):
                        if tuned_container is not None:
                            stop_containers_by_name([tuned_container.name])

                    console.print(
                        "[yellow]⚠️ Échec de la relance optimisée, restauration du ratio précédent "
                        f"({previous_ratio:.3f}). Détail: {tune_exc}[/yellow]"
                    )

                    with console.status("[cyan]Restauration de la configuration précédente...[/cyan]", spinner="dots"):
                        container = run_model_container(
                            model_id=model_id,
                            host_port=port,
                            gpu_index=gpu_index,
                            gpu_utilization=previous_ratio,
                            max_num_seqs=resolved_max_num_seqs,
                            max_model_len=resolved_max_model_len,
                            tool_call_parser=resolved_tool_call_parser,
                            reasoning_parser=resolved_reasoning_parser,
                            enable_thinking=resolved_enable_thinking,
                            required_ratio=new_required_ratio,
                            allow_long_context_override=resolved_allow_long_context_override,
                            config=CONFIG,
                        )

                    wait_until_model_ready(console, container.name, timeout_seconds=startup_timeout)
                    selected_ratio = previous_ratio

        console.print("[bold green]✅ Modèle lancé[/bold green]")
        console.print(f"Conteneur: [cyan]{container.name}[/cyan]")
        console.print(f"API vLLM: [cyan]http://localhost:{port}[/cyan]")
        console.print(f"GPU: [cyan]{gpu_index}[/cyan] | Utilization: [cyan]{selected_ratio:.3f}[/cyan]")
        console.print(f"Logs: [cyan]docker logs -f {container.name}[/cyan]")
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)
