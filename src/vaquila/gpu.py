"""Services GPU basés sur NVML."""

from __future__ import annotations

from contextlib import suppress

from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
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


def read_all_gpu_snapshots() -> dict[int, GpuSnapshot]:
    """Lit les métriques mémoire pour tous les GPUs NVIDIA détectés."""
    try:
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        if gpu_count <= 0:
            raise VaquilaError("Aucun GPU NVIDIA détecté via NVML.")

        snapshots: dict[int, GpuSnapshot] = {}
        for index in range(gpu_count):
            handle = nvmlDeviceGetHandleByIndex(index)
            memory = nvmlDeviceGetMemoryInfo(handle)
            snapshots[index] = GpuSnapshot(
                index=index,
                total_bytes=int(memory.total),
                free_bytes=int(memory.free),
                used_bytes=int(memory.used),
            )

        return snapshots
    except NVMLError as exc:
        raise VaquilaError(
            "NVIDIA NVML indisponible. Vérifie les drivers et la présence d'un GPU NVIDIA."
        ) from exc
    finally:
        with suppress(NVMLError):
            nvmlShutdown()


def compute_gpu_memory_utilization(snapshot: GpuSnapshot, security_buffer_gb: float) -> float:
    """Calcule dynamiquement le ratio vLLM en réservant un buffer mémoire système adapté à l'OS."""
    if security_buffer_gb <= 0:
        raise VaquilaError("Le buffer VRAM doit être supérieur à 0 Go.")

    buffer_bytes = int(security_buffer_gb * (1024**3))
    usable_bytes = snapshot.free_bytes - buffer_bytes

    if usable_bytes <= 0:
        free_gb = _bytes_to_gb(snapshot.free_bytes)
        raise VaquilaError(
            f"VRAM insuffisante: {free_gb:.2f} Gio libres, buffer demandé {security_buffer_gb:.2f} Gio. "
            "Ajuste le buffer ou libère de la mémoire GPU."
        )

    raw_ratio = usable_bytes / snapshot.total_bytes
    # Toujours borner le ratio pour éviter les crashs vLLM
    return max(0.10, min(raw_ratio, 0.98))


def compute_adaptive_gpu_memory_utilization(
    snapshot: GpuSnapshot,
    security_buffer_gb: float,
    minimum_buffer_gb: float = 0.25,
) -> tuple[float, float]:
    """Calcule un ratio vLLM adaptatif quand la VRAM est fragmentée par plusieurs modèles.

    Retourne un tuple (ratio, buffer_effectif_gb).
    """
    if security_buffer_gb <= 0:
        raise VaquilaError("Le buffer VRAM doit être supérieur à 0 Go.")

    if minimum_buffer_gb <= 0:
        raise VaquilaError("Le buffer minimal VRAM doit être supérieur à 0 Go.")

    min_buffer_bytes = int(minimum_buffer_gb * (1024**3))
    desired_buffer_bytes = int(security_buffer_gb * (1024**3))

    if snapshot.free_bytes <= min_buffer_bytes:
        free_gb = _bytes_to_gb(snapshot.free_bytes)
        raise VaquilaError(
            f"VRAM insuffisante: {free_gb:.2f} Gio libres, minimum requis {minimum_buffer_gb:.2f} Gio. "
            "Libère de la mémoire GPU avant de lancer un second modèle."
        )

    effective_buffer_bytes = min(desired_buffer_bytes, max(min_buffer_bytes, int(snapshot.free_bytes * 0.15)))

    usable_bytes = snapshot.free_bytes - effective_buffer_bytes
    if usable_bytes <= 0:
        free_gb = _bytes_to_gb(snapshot.free_bytes)
        raise VaquilaError(
            f"VRAM insuffisante après ajustement adaptatif: {free_gb:.2f} Gio libres."
        )

    # Garde une petite marge absolue pour éviter de frôler la VRAM libre réelle.
    hard_reserve_bytes = 64 * 1024 * 1024
    max_ratio_by_free = max(0.0, (snapshot.free_bytes - hard_reserve_bytes) / snapshot.total_bytes)
    raw_ratio = usable_bytes / snapshot.total_bytes

    ratio = min(raw_ratio, max_ratio_by_free, 0.98)
    ratio = max(0.01, ratio)

    return ratio, _bytes_to_gb(effective_buffer_bytes)
