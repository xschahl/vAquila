"""Helpers de résolution runtime et heuristiques de ratio GPU."""

from __future__ import annotations

import typer

from vaquila.exceptions import VaquilaError


def normalize_optional_text(value: str | None) -> str | None:
    """Normalise une chaîne optionnelle (vide -> None)."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def estimate_required_ratio(
    max_num_seqs: int,
    max_model_len: int,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool,
) -> float:
    """Estime un ratio GPU minimal requis selon la configuration runtime demandée."""
    context_factor = max_model_len / 16384
    ratio = 0.02
    ratio += 0.001 * max_num_seqs
    ratio += 0.02 * context_factor
    if tool_call_parser:
        ratio += 0.002
    if reasoning_parser:
        ratio += 0.002
    if enable_thinking:
        ratio += 0.001
    return max(0.02, min(ratio, 0.90))


def is_retryable_vram_error(message: str) -> bool:
    """Détecte une erreur vLLM qui justifie une montée de ratio."""
    patterns = (
        "No available memory for the cache blocks",
        "Free memory on device",
        "less than desired GPU memory utilization",
    )
    return any(pattern in message for pattern in patterns)


def ratio_candidates(min_ratio: float, max_ratio: float) -> list[float]:
    """Génère des paliers de ratio entre minimum requis et maximum disponible."""
    if min_ratio > max_ratio:
        return []

    candidates = [round(min_ratio, 3)]
    steps = (0.03, 0.05, 0.08, 0.12, 0.18)
    for step in steps:
        candidate = round(min_ratio + step, 3)
        if candidate < max_ratio and candidate not in candidates:
            candidates.append(candidate)

    max_rounded = round(max_ratio, 3)
    if max_rounded not in candidates:
        candidates.append(max_rounded)

    return candidates


def resolve_run_runtime_settings(
    max_num_seqs: int | None,
    max_model_len: int | None,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool | None,
) -> tuple[int, int, str | None, str | None, bool]:
    """Résout les paramètres runtime via options ou prompts interactifs."""
    resolved_max_num_seqs = max_num_seqs
    if resolved_max_num_seqs is None:
        resolved_max_num_seqs = typer.prompt("Nombre de requêtes en parallèle (max-num-seqs)", default=1, type=int)

    resolved_max_model_len = max_model_len
    if resolved_max_model_len is None:
        resolved_max_model_len = typer.prompt("Contexte par utilisateur (max-model-len)", default=16384, type=int)

    resolved_tool_call_parser = normalize_optional_text(tool_call_parser)
    if tool_call_parser is None:
        prompted_tool_call_parser = typer.prompt(
            "Tool call parser (laisser vide = aucun)",
            default="",
            show_default=False,
        )
        resolved_tool_call_parser = normalize_optional_text(prompted_tool_call_parser)

    resolved_reasoning_parser = normalize_optional_text(reasoning_parser)
    if reasoning_parser is None:
        prompted_reasoning_parser = typer.prompt(
            "Reasoning parser (laisser vide = aucun)",
            default="",
            show_default=False,
        )
        resolved_reasoning_parser = normalize_optional_text(prompted_reasoning_parser)

    resolved_enable_thinking = enable_thinking
    if resolved_enable_thinking is None:
        resolved_enable_thinking = typer.confirm("Activer le thinking ?", default=True)

    if resolved_max_num_seqs <= 0:
        raise VaquilaError("max-num-seqs doit être supérieur à 0.")
    if resolved_max_model_len <= 0:
        raise VaquilaError("max-model-len doit être supérieur à 0.")

    return (
        resolved_max_num_seqs,
        resolved_max_model_len,
        resolved_tool_call_parser,
        resolved_reasoning_parser,
        resolved_enable_thinking,
    )
