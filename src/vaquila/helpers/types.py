"""Shared type definitions for the CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchPlan:
    """vLLM launch plan for a model."""

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
