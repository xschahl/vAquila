"""NVML-based GPU services."""

from __future__ import annotations

from contextlib import suppress

from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
    nvmlShutdown,
)

from vaquila.exceptions import VaquilaError
from vaquila.models import GpuSnapshot


def _bytes_to_gb(value: int) -> float:
    """Convert a value in bytes to GiB."""
    return value / (1024**3)


def read_gpu_snapshot(index: int = 0) -> GpuSnapshot:
    """Read NVIDIA GPU memory metrics through NVML."""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(index)
        memory = nvmlDeviceGetMemoryInfo(handle)
        raw_name = nvmlDeviceGetName(handle)
        gpu_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        return GpuSnapshot(
            index=index,
            name=gpu_name,
            total_bytes=int(memory.total),
            free_bytes=int(memory.free),
            used_bytes=int(memory.used),
        )
    except NVMLError as exc:
        raise VaquilaError(
            "NVIDIA NVML is unavailable. Verify drivers and the presence of an NVIDIA GPU."
        ) from exc
    finally:
        with suppress(NVMLError):
            nvmlShutdown()


def read_all_gpu_snapshots() -> dict[int, GpuSnapshot]:
    """Read memory metrics for all detected NVIDIA GPUs."""
    try:
        nvmlInit()
        gpu_count = nvmlDeviceGetCount()
        if gpu_count <= 0:
            raise VaquilaError("No NVIDIA GPU detected through NVML.")

        snapshots: dict[int, GpuSnapshot] = {}
        for index in range(gpu_count):
            handle = nvmlDeviceGetHandleByIndex(index)
            memory = nvmlDeviceGetMemoryInfo(handle)
            raw_name = nvmlDeviceGetName(handle)
            gpu_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
            snapshots[index] = GpuSnapshot(
                index=index,
                name=gpu_name,
                total_bytes=int(memory.total),
                free_bytes=int(memory.free),
                used_bytes=int(memory.used),
            )

        return snapshots
    except NVMLError as exc:
        raise VaquilaError(
            "NVIDIA NVML is unavailable. Verify drivers and the presence of an NVIDIA GPU."
        ) from exc
    finally:
        with suppress(NVMLError):
            nvmlShutdown()


def compute_gpu_memory_utilization(snapshot: GpuSnapshot, security_buffer_gb: float) -> float:
    """Compute the vLLM ratio dynamically while reserving a system memory safety buffer."""
    if security_buffer_gb <= 0:
        raise VaquilaError("The VRAM buffer must be greater than 0 GiB.")

    buffer_bytes = int(security_buffer_gb * (1024**3))
    usable_bytes = snapshot.free_bytes - buffer_bytes

    if usable_bytes <= 0:
        free_gb = _bytes_to_gb(snapshot.free_bytes)
        raise VaquilaError(
            f"Insufficient VRAM: {free_gb:.2f} GiB free, requested buffer {security_buffer_gb:.2f} GiB. "
            "Adjust the buffer or free GPU memory."
        )

    raw_ratio = usable_bytes / snapshot.total_bytes
    # Always bound the ratio to prevent vLLM crashes.
    return max(0.10, min(raw_ratio, 0.98))


def compute_adaptive_gpu_memory_utilization(
    snapshot: GpuSnapshot,
    security_buffer_gb: float,
    minimum_buffer_gb: float = 0.25,
) -> tuple[float, float]:
    """Compute an adaptive vLLM ratio when VRAM is fragmented across multiple models.

    Return a tuple ``(ratio, effective_buffer_gb)``.
    """
    if security_buffer_gb <= 0:
        raise VaquilaError("The VRAM buffer must be greater than 0 GiB.")

    if minimum_buffer_gb <= 0:
        raise VaquilaError("The minimum VRAM buffer must be greater than 0 GiB.")

    min_buffer_bytes = int(minimum_buffer_gb * (1024**3))
    desired_buffer_bytes = int(security_buffer_gb * (1024**3))

    if snapshot.free_bytes <= min_buffer_bytes:
        free_gb = _bytes_to_gb(snapshot.free_bytes)
        raise VaquilaError(
            f"Insufficient VRAM: {free_gb:.2f} GiB free, minimum required {minimum_buffer_gb:.2f} GiB. "
            "Free GPU memory before launching a second model."
        )

    effective_buffer_bytes = min(desired_buffer_bytes, max(min_buffer_bytes, int(snapshot.free_bytes * 0.15)))

    usable_bytes = snapshot.free_bytes - effective_buffer_bytes
    if usable_bytes <= 0:
        free_gb = _bytes_to_gb(snapshot.free_bytes)
        raise VaquilaError(
            f"Insufficient VRAM after adaptive adjustment: {free_gb:.2f} GiB free."
        )

    # Keep a small absolute margin to avoid getting too close to real free VRAM.
    hard_reserve_bytes = 64 * 1024 * 1024
    max_ratio_by_free = max(0.0, (snapshot.free_bytes - hard_reserve_bytes) / snapshot.total_bytes)
    raw_ratio = usable_bytes / snapshot.total_bytes

    ratio = min(raw_ratio, max_ratio_by_free, 0.98)
    ratio = max(0.01, ratio)

    return ratio, _bytes_to_gb(effective_buffer_bytes)
