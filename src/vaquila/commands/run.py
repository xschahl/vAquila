"""Model launch command (`run`)."""

from __future__ import annotations

from contextlib import suppress
import json
import platform
from pathlib import Path

import typer
from rich.console import Console

from vaquila.cli_helpers import (
    estimate_required_ratio,
    extract_kv_cache_memory_bounds,
    extract_kv_max_concurrency,
    is_retryable_vram_error,
    ratio_candidates,
    resolve_context_strategy,
    resolve_kv_cache_dtype,
    resolve_quantization_strategy,
    resolve_run_runtime_settings,
    suggest_ratio_from_kv_cache_error,
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


def _tuning_hints_path() -> Path:
    """Return the persistence file for optimized ratio hints."""
    return CONFIG.hf_cache_host_path / "vaquila_tuning_hints.json"


def _build_tuning_hint_key(
    model_id: str,
    max_num_seqs: int,
    max_model_len: int,
    quantization: str | None,
    kv_cache_dtype: str,
) -> str:
    """Build a stable tuning key for a given launch context."""
    quant_label = quantization or "none"
    return f"{model_id}|seqs={max_num_seqs}|ctx={max_model_len}|quant={quant_label}|kv={kv_cache_dtype}"


def _load_tuning_hint_ratio(key: str) -> float | None:
    """Load a persisted recommended ratio when available."""
    path = _tuning_hints_path()
    with suppress(OSError, json.JSONDecodeError, TypeError, ValueError):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw = payload.get(key)
            if isinstance(raw, (int, float)):
                value = float(raw)
                if 0.0 < value <= 1.0:
                    return value
    return None


def _save_tuning_hint_ratio(key: str, ratio: float) -> None:
    """Persist the last stable ratio to speed up future launches."""
    path = _tuning_hints_path()
    with suppress(OSError, json.JSONDecodeError, TypeError):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, float] = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload = {k: float(v) for k, v in existing.items() if isinstance(v, (int, float))}
        payload[key] = round(ratio, 3)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_run(
    model_id: str,
    port: int,
    gpu_index: int,
    buffer_gb: float | None,
    startup_timeout: int,
    max_num_seqs: int | None,
    max_model_len: int | None,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool | None,
    allow_long_context_override: bool | None,
    quantization: str,
    kv_cache_dtype: str | None,
) -> None:
    """Launch a model in a background vLLM container."""
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

        resolved_quantization, quantization_label = resolve_quantization_strategy(
            model_id=model_id,
            quantization=quantization,
        )
        resolved_kv_cache_dtype = resolve_kv_cache_dtype(kv_cache_dtype)

        snapshot = read_gpu_snapshot(gpu_index)
        total_vram_gb = snapshot.total_bytes / (1024**3)

        new_required_ratio = estimate_required_ratio(
            max_num_seqs=resolved_max_num_seqs,
            max_model_len=resolved_max_model_len,
            tool_call_parser=resolved_tool_call_parser,
            reasoning_parser=resolved_reasoning_parser,
            enable_thinking=resolved_enable_thinking,
            kv_cache_dtype=resolved_kv_cache_dtype,
            quantization=resolved_quantization,
            model_id=model_id,
            total_vram_gb=total_vram_gb,
        )

        running_on_same_gpu = [
            item for item in list_managed_containers() if item.gpu_index == gpu_index and item.status == "running"
        ]

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
                "[yellow]⚠️ VRAM is fragmented by already running models: "
                f"buffer adjusted to {adaptive_buffer:.2f} GiB, max available ratio={max_available_ratio:.3f}.[/yellow]"
            )

        if max_available_ratio < new_required_ratio:
            raise VaquilaError(
                "Runtime configuration is too demanding for available VRAM before vLLM startup. "
                f"max_available_ratio={max_available_ratio:.3f}, required_ratio={new_required_ratio:.3f}. "
                "Reduce max-num-seqs/max-model-len, or free VRAM (stop containers with `vaq ps` then `vaq stop <model_id>`)."
            )

        ratio = max(new_required_ratio, 0.02)
        tuning_key = _build_tuning_hint_key(
            model_id=model_id,
            max_num_seqs=resolved_max_num_seqs,
            max_model_len=resolved_max_model_len,
            quantization=resolved_quantization,
            kv_cache_dtype=resolved_kv_cache_dtype,
        )
        hinted_ratio = _load_tuning_hint_ratio(tuning_key)
        initial_ratio = ratio
        if hinted_ratio is not None and ratio <= hinted_ratio <= max_available_ratio:
            initial_ratio = hinted_ratio
            console.print(
                f"[cyan]Tuning hint detected:[/cyan] previous stable ratio={hinted_ratio:.3f} "
                "(reused to speed up startup)."
            )

        typer.secho(
            f"[vAquila] VRAM safety buffer: {buffer:.2f} GiB | Applied base ratio: {ratio:.3f} | Max available ratio: {max_available_ratio:.3f}",
            fg=typer.colors.CYAN,
        )
        console.print(
            "[cyan]Requested runtime:[/cyan] "
            f"max_num_seqs={resolved_max_num_seqs}, max_model_len={resolved_max_model_len}, "
            f"tool_call_parser={resolved_tool_call_parser or 'none'}, "
            f"reasoning_parser={resolved_reasoning_parser or 'none'}, "
            f"enable_thinking={resolved_enable_thinking}, quantization={quantization_label}, "
            f"kv_cache_dtype={resolved_kv_cache_dtype}, required_ratio~{new_required_ratio:.3f}"
        )

        container = None
        selected_ratio = initial_ratio
        attempt_ratios = ratio_candidates(min_ratio=initial_ratio, max_ratio=max_available_ratio)
        if not attempt_ratios:
            raise VaquilaError(
                "Unable to compute a valid startup ratio. "
                f"min={initial_ratio:.3f}, max={max_available_ratio:.3f}"
            )
        planned_attempt_ratios = list(attempt_ratios)

        attempt_index = 0
        while attempt_index < len(planned_attempt_ratios):
            attempt_ratio = planned_attempt_ratios[attempt_index]
            selected_ratio = attempt_ratio
            with console.status(
                "[cyan]Creating vLLM container and preparing Hugging Face download...[/cyan]",
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
                    quantization=resolved_quantization,
                    kv_cache_dtype=resolved_kv_cache_dtype,
                    config=CONFIG,
                )

            try:
                wait_until_model_ready(console, container.name, timeout_seconds=startup_timeout)
                break
            except VaquilaError as exc:
                with suppress(VaquilaError):
                    stop_containers_by_name([container.name])

                error_text = str(exc)
                retryable = is_retryable_vram_error(error_text)
                inserted_ratio = None

                if retryable:
                    suggested_ratio = suggest_ratio_from_kv_cache_error(attempt_ratio, error_text)
                    if suggested_ratio is not None:
                        next_ratio = min(max_available_ratio, round(suggested_ratio, 3))
                        if next_ratio > attempt_ratio:
                            if all(abs(next_ratio - planned) >= 0.001 for planned in planned_attempt_ratios):
                                planned_attempt_ratios.append(next_ratio)
                                planned_attempt_ratios = sorted(planned_attempt_ratios)
                                inserted_ratio = next_ratio

                        bounds = extract_kv_cache_memory_bounds(error_text)
                        if bounds is not None and inserted_ratio is not None:
                            needed_gib, available_gib = bounds
                            console.print(
                                "[yellow]ℹ️ Precise adjustment from vLLM logs: "
                                f"needed={needed_gib:.2f} GiB, available={available_gib:.2f} GiB, "
                                f"suggested_ratio={inserted_ratio:.3f}.[/yellow]"
                            )

                    max_ratio_rounded = round(max_available_ratio, 3)
                    if (
                        attempt_ratio < max_available_ratio
                        and all(abs(max_ratio_rounded - planned) >= 0.001 for planned in planned_attempt_ratios)
                    ):
                        planned_attempt_ratios.append(max_ratio_rounded)
                        planned_attempt_ratios = sorted(planned_attempt_ratios)

                next_index = attempt_index + 1
                if not retryable or next_index >= len(planned_attempt_ratios):
                    raise

                next_ratio = planned_attempt_ratios[next_index]
                console.print(
                    "[yellow]⚠️ Insufficient ratio for KV cache, next attempt: "
                    f"{next_ratio:.3f} (previous={attempt_ratio:.3f})[/yellow]"
                )
                attempt_index += 1
                continue

            attempt_index += 1

        def _read_observed_concurrency(container_name: str) -> float | None:
            runtime_container = get_container(container_name)
            runtime_logs = runtime_container.logs(tail=600).decode("utf-8", errors="replace")
            return extract_kv_max_concurrency(runtime_logs, resolved_max_model_len)

        observed_concurrency = _read_observed_concurrency(container.name)
        requested_concurrency = max(1.0, float(resolved_max_num_seqs))
        target_concurrency = max(requested_concurrency, float(resolved_max_num_seqs) * 1.10)
        lower_concurrency_bound = requested_concurrency
        upper_concurrency_bound = max(lower_concurrency_bound + 0.45, target_concurrency * 1.15)

        soft_floor_ratio = max(0.02, round(new_required_ratio * 0.55, 3))
        floor_ratio = min(selected_ratio, soft_floor_ratio)
        max_tuning_attempts = 5
        best_ratio = selected_ratio
        best_observed_concurrency = observed_concurrency
        lowest_failed_ratio: float | None = None
        highest_failed_ratio: float | None = None
        attempted_tuning_ratios: set[float] = set()

        for tune_index in range(1, max_tuning_attempts + 1):
            if best_observed_concurrency is None:
                break
            if lower_concurrency_bound <= best_observed_concurrency <= upper_concurrency_bound:
                break

            remaining_headroom = best_ratio - floor_ratio

            if best_observed_concurrency < lower_concurrency_bound:
                if highest_failed_ratio is not None and highest_failed_ratio > best_ratio:
                    next_ratio = round((best_ratio + highest_failed_ratio) / 2, 3)
                else:
                    growth_factor = min(
                        1.35,
                        max(1.08, lower_concurrency_bound / max(best_observed_concurrency, 0.10)),
                    )
                    next_ratio = round(min(max_available_ratio, best_ratio * growth_factor), 3)
                    next_ratio = max(next_ratio, round(best_ratio + 0.003, 3))
            else:
                if remaining_headroom < 0.004:
                    console.print(
                        "[yellow]⚠️ KV cache is still oversized, but ratio is already near safety floor "
                        f"({best_ratio:.3f}).[/yellow]"
                    )
                    break

                if lowest_failed_ratio is not None and lowest_failed_ratio < best_ratio:
                    next_ratio = round((best_ratio + lowest_failed_ratio) / 2, 3)
                else:
                    ratio_by_observed = best_ratio * (target_concurrency / best_observed_concurrency) * 1.03
                    ratio_by_headroom = best_ratio - (remaining_headroom * 0.60)
                    next_ratio = round(max(floor_ratio, ratio_by_observed, ratio_by_headroom), 3)
                    next_ratio = min(next_ratio, round(best_ratio - 0.003, 3))
                    next_ratio = max(next_ratio, round(best_ratio * 0.70, 3))

            if next_ratio <= 0:
                break
            if abs(next_ratio - best_ratio) < 0.001:
                break
            if next_ratio > max_available_ratio:
                next_ratio = round(max_available_ratio, 3)
            if next_ratio in attempted_tuning_ratios:
                console.print(
                    "[yellow]⚠️ Ratio already tested in this tuning pass; stopping to avoid loop "
                    f"({next_ratio:.3f}).[/yellow]"
                )
                break
            attempted_tuning_ratios.add(next_ratio)

            previous_ratio = best_ratio
            previous_container_name = container.name
            console.print(
                "[yellow]⚠️ KV cache adjustment: "
                f"observed_concurrency={best_observed_concurrency:.2f}x, "
                f"target=[{lower_concurrency_bound:.2f}..{upper_concurrency_bound:.2f}]x. "
                f"Adjustment {tune_index}/{max_tuning_attempts}: {previous_ratio:.3f} -> {next_ratio:.3f}[/yellow]"
            )

            with suppress(VaquilaError):
                stop_containers_by_name([previous_container_name])

            tuned_container = None
            try:
                with console.status("[cyan]Restarting with adjusted ratio...[/cyan]", spinner="dots"):
                    tuned_container = run_model_container(
                        model_id=model_id,
                        host_port=port,
                        gpu_index=gpu_index,
                        gpu_utilization=next_ratio,
                        max_num_seqs=resolved_max_num_seqs,
                        max_model_len=resolved_max_model_len,
                        tool_call_parser=resolved_tool_call_parser,
                        reasoning_parser=resolved_reasoning_parser,
                        enable_thinking=resolved_enable_thinking,
                        required_ratio=new_required_ratio,
                        allow_long_context_override=resolved_allow_long_context_override,
                        quantization=resolved_quantization,
                        kv_cache_dtype=resolved_kv_cache_dtype,
                        config=CONFIG,
                    )

                wait_until_model_ready(console, tuned_container.name, timeout_seconds=startup_timeout)
                container = tuned_container
                candidate_observed_concurrency = _read_observed_concurrency(container.name)

                if candidate_observed_concurrency is None:
                    best_ratio = next_ratio
                    best_observed_concurrency = candidate_observed_concurrency
                    continue

                if candidate_observed_concurrency < lower_concurrency_bound:
                    # This ratio is too low to satisfy max-num-seqs: lower bound for the search window.
                    lowest_failed_ratio = next_ratio
                    with suppress(VaquilaError):
                        stop_containers_by_name([container.name])

                    with console.status(
                        "[cyan]Restoring stable ratio (concurrency too low)...[/cyan]",
                        spinner="dots",
                    ):
                        container = run_model_container(
                            model_id=model_id,
                            host_port=port,
                            gpu_index=gpu_index,
                            gpu_utilization=best_ratio,
                            max_num_seqs=resolved_max_num_seqs,
                            max_model_len=resolved_max_model_len,
                            tool_call_parser=resolved_tool_call_parser,
                            reasoning_parser=resolved_reasoning_parser,
                            enable_thinking=resolved_enable_thinking,
                            required_ratio=new_required_ratio,
                            allow_long_context_override=resolved_allow_long_context_override,
                            quantization=resolved_quantization,
                            kv_cache_dtype=resolved_kv_cache_dtype,
                            config=CONFIG,
                        )

                    wait_until_model_ready(console, container.name, timeout_seconds=startup_timeout)
                    continue

                best_ratio = next_ratio
                best_observed_concurrency = candidate_observed_concurrency
            except VaquilaError as tune_exc:
                tune_error_text = str(tune_exc)
                if next_ratio < previous_ratio:
                    lowest_failed_ratio = next_ratio
                else:
                    highest_failed_ratio = next_ratio

                kv_suggested_ratio = suggest_ratio_from_kv_cache_error(next_ratio, tune_error_text)
                if kv_suggested_ratio is not None:
                    bounded_kv_ratio = min(max_available_ratio, kv_suggested_ratio)
                    if bounded_kv_ratio > next_ratio:
                        lowered_bound = min(round(previous_ratio - 0.001, 3), round(bounded_kv_ratio, 3))
                        if lowered_bound > next_ratio:
                            lowest_failed_ratio = max(lowest_failed_ratio or 0.0, lowered_bound)
                            console.print(
                                "[yellow]ℹ️ Lower-bound adjustment from KV cache memory: "
                                f"suggested_min_ratio~{lowered_bound:.3f}[/yellow]"
                            )

                with suppress(VaquilaError):
                    if tuned_container is not None:
                        stop_containers_by_name([tuned_container.name])

                console.print(
                    "[yellow]⚠️ Tuning attempt failed, restoring stable configuration "
                    f"({previous_ratio:.3f}). Details: {tune_error_text}[/yellow]"
                )

                with console.status("[cyan]Restoring previous configuration...[/cyan]", spinner="dots"):
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
                        quantization=resolved_quantization,
                        kv_cache_dtype=resolved_kv_cache_dtype,
                        config=CONFIG,
                    )

                wait_until_model_ready(console, container.name, timeout_seconds=startup_timeout)
                best_ratio = previous_ratio
                best_observed_concurrency = _read_observed_concurrency(container.name)

                if lowest_failed_ratio is not None and lowest_failed_ratio >= best_ratio:
                    console.print(
                        "[yellow]⚠️ KV lower bound is too close to stable ratio; stopping tuning "
                        f"(stable={best_ratio:.3f}, min_kv={lowest_failed_ratio:.3f}).[/yellow]"
                    )
                    break

                if (
                    lowest_failed_ratio is not None
                    and lowest_failed_ratio < best_ratio
                    and (best_ratio - lowest_failed_ratio) <= 0.001
                ):
                    console.print(
                        "[yellow]⚠️ Near-minimal stable ratio reached: cannot reduce further without failure "
                        f"(stable={best_ratio:.3f}, fail={lowest_failed_ratio:.3f}).[/yellow]"
                    )
                    break

            if (
                lowest_failed_ratio is not None
                and lowest_failed_ratio < best_ratio
                and (best_ratio - lowest_failed_ratio) <= 0.001
                and best_observed_concurrency is not None
                and best_observed_concurrency >= lower_concurrency_bound
            ):
                console.print(
                    "[yellow]⚠️ Stable/fail window is very narrow; stopping tuning "
                    f"(stable={best_ratio:.3f}, fail={lowest_failed_ratio:.3f}).[/yellow]"
                )
                break

        selected_ratio = best_ratio
        observed_concurrency = best_observed_concurrency
        _save_tuning_hint_ratio(tuning_key, selected_ratio)

        console.print("[bold green]✅ Model launched[/bold green]")
        console.print(f"Container: [cyan]{container.name}[/cyan]")
        console.print(f"API vLLM: [cyan]http://localhost:{port}[/cyan]")
        console.print(f"GPU: [cyan]{gpu_index}[/cyan] | Utilization: [cyan]{selected_ratio:.3f}[/cyan]")
        if observed_concurrency is not None:
            console.print(
                f"Observed KV max concurrency (context={resolved_max_model_len}): [cyan]{observed_concurrency:.2f}x[/cyan]"
            )
        console.print(f"Logs: [cyan]docker logs -f {container.name}[/cyan]")
    except VaquilaError as exc:
        console.print(f"[bold red]❌ {exc}[/bold red]")
        raise typer.Exit(code=1)
