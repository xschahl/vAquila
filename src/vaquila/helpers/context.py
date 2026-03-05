"""Helpers de stratégie de contexte modèle."""

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
    """Résout la stratégie si le contexte demandé dépasse la limite du modèle."""
    model_context_limit = resolve_model_context_limit(model_id)
    if model_context_limit is None or requested_max_model_len <= model_context_limit:
        return requested_max_model_len, bool(allow_long_context_override)

    console.print(
        "[yellow]⚠️ Le contexte demandé dépasse la limite du modèle:[/yellow] "
        f"demandé={requested_max_model_len}, limite modèle={model_context_limit}."
    )

    if allow_long_context_override is True:
        console.print(
            "[yellow]Mode override activé: vLLM sera lancé avec VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "
            "(risque de comportement instable selon le modèle).[/yellow]"
        )
        return requested_max_model_len, True

    if allow_long_context_override is False:
        console.print(f"[cyan]Contexte ajusté à la limite modèle: {model_context_limit}.[/cyan]")
        return model_context_limit, False

    choice = typer.prompt(
        "Choix [1=optimisation override (risqué), 2=utiliser limite modèle, 3=annuler]",
        default="2",
    ).strip()

    if choice == "1":
        console.print(
            "[yellow]Mode override activé: vLLM sera lancé avec VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "
            "(risque de comportement instable selon le modèle).[/yellow]"
        )
        return requested_max_model_len, True

    if choice == "2":
        console.print(f"[cyan]Contexte ajusté à la limite modèle: {model_context_limit}.[/cyan]")
        return model_context_limit, False

    raise VaquilaError("Lancement annulé par l'utilisateur (stratégie contexte).")
