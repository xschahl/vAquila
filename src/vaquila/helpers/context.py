"""Model context strategy helpers."""

from __future__ import annotations

import typer
from rich.console import Console

from vaquila.exceptions import VaquilaError
from vaquila.helpers.cache import resolve_model_context_limit


def resolve_context_strategy(
    console: Console,
    model_id: str,
    requested_max_model_len: int,
    allow_long_context_override: bool | None,
) -> tuple[int, bool]:
    """Resolve strategy when requested context exceeds the model limit."""
    model_context_limit = resolve_model_context_limit(model_id)
    if model_context_limit is None or requested_max_model_len <= model_context_limit:
        return requested_max_model_len, bool(allow_long_context_override)

    console.print(
        "[yellow]⚠️ Requested context exceeds model limit:[/yellow] "
        f"requested={requested_max_model_len}, model_limit={model_context_limit}."
    )

    if allow_long_context_override is True:
        console.print(
            "[yellow]Override mode enabled: vLLM will run with VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "
            "(model-specific instability risk).[/yellow]"
        )
        return requested_max_model_len, True

    if allow_long_context_override is False:
        console.print(f"[cyan]Context clamped to model limit: {model_context_limit}.[/cyan]")
        return model_context_limit, False

    choice = typer.prompt(
        "Choose [1=override optimization (risky), 2=use model limit, 3=cancel]",
        default="2",
    ).strip()

    if choice == "1":
        console.print(
            "[yellow]Override mode enabled: vLLM will run with VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "
            "(model-specific instability risk).[/yellow]"
        )
        return requested_max_model_len, True

    if choice == "2":
        console.print(f"[cyan]Context clamped to model limit: {model_context_limit}.[/cyan]")
        return model_context_limit, False

    raise VaquilaError("Launch canceled by user (context strategy).")
