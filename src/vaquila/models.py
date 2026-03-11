"""Internal data models for orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSnapshot:
    """Memory snapshot for an NVIDIA GPU."""

    index: int
    name: str | None
    total_bytes: int
    free_bytes: int
    used_bytes: int


@dataclass(frozen=True)
class ManagedContainer:
    """vAquila container projection displayed by the `ps` command."""

    name: str
    model_id: str
    status: str
    host_port: int | None
    compute_backend: str | None
    gpu_index: int | None
    gpu_used_bytes: int | None
    gpu_utilization: float | None
    cpu_utilization: float | None
    cpu_kv_cache_space: str | None
    max_num_seqs: int | None
    max_model_len: int | None
    tool_call_parser: str | None
    reasoning_parser: str | None
    enable_thinking: bool | None
    required_ratio: float | None
    allow_long_context_override: bool | None
    instance_id: str | None = None
