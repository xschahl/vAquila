"""Types partagés pour la CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchPlan:
    """Plan de lancement vLLM pour un modèle."""

    model_id: str
    host_port: int
    existing_name: str | None
    max_num_seqs: int
    max_model_len: int
    tool_call_parser: str | None
    reasoning_parser: str | None
    enable_thinking: bool
    required_ratio: float
    allow_long_context_override: bool
