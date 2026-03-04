"""Services GPU basés sur NVML."""

from __future__ import annotations

from contextlib import suppress

from pynvml import (
    NVMLError,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)

from vaquila.exceptions import VaquilaError
from vaquila.models import GpuSnapshot


def _bytes_to_gb(value: int) -> float:
    """Convertit une valeur en octets vers des Gio."""
    return value / (1024**3)


def read_gpu_snapshot(index: int = 0) -> GpuSnapshot:
    """Lit les métriques mémoire d'un GPU NVIDIA via NVML."""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(index)
        memory = nvmlDeviceGetMemoryInfo(handle)
        return GpuSnapshot(
            index=index,
            total_bytes=int(memory.total),
            free_bytes=int(memory.free),
            used_bytes=int(memory.used),
        )
    except NVMLError as exc:
        raise VaquilaError(
            "NVIDIA NVML indisponible. Vérifie les drivers et la présence d'un GPU NVIDIA."
        ) from exc
    finally:
        with suppress(NVMLError):
            nvmlShutdown()


def compute_gpu_memory_utilization(snapshot: GpuSnapshot, security_buffer_gb: float) -> float:
    """Calcule un ratio vLLM sûr en réservant un buffer mémoire système."""
    if security_buffer_gb <= 0:
        raise VaquilaError("Le buffer VRAM doit être supérieur à 0 Go.")

    buffer_bytes = int(security_buffer_gb * (1024**3))
    usable_bytes = snapshot.free_bytes - buffer_bytes

    if usable_bytes <= 0:
        free_gb = _bytes_to_gb(snapshot.free_bytes)
        raise VaquilaError(
            f"VRAM insuffisante: {free_gb:.2f} Gio libres, buffer demandé {security_buffer_gb:.2f} Gio."
        )

    raw_ratio = usable_bytes / snapshot.total_bytes
    return max(0.10, min(raw_ratio, 0.98))
