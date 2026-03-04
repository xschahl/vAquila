"""Modèles de données internes pour l'orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSnapshot:
    """Snapshot mémoire d'un GPU NVIDIA."""

    index: int
    total_bytes: int
    free_bytes: int
    used_bytes: int


@dataclass(frozen=True)
class ManagedContainer:
    """Projection d'un conteneur vAquila affichée par la commande `ps`."""

    name: str
    model_id: str
    status: str
    host_port: int | None
    gpu_index: int | None
    gpu_used_bytes: int | None
