"""Docker services for vLLM container orchestration."""

from __future__ import annotations

import os
from pathlib import Path
import re

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container
from docker.types import DeviceRequest

from vaquila.config import RuntimeConfig
from vaquila.exceptions import VaquilaError
from vaquila.models import GpuSnapshot, ManagedContainer


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize a model ID to build a readable container name."""
    return (
        model_id.lower()
        .replace("/", "-")
        .replace("_", "-")
        .replace(".", "-")
        .replace(":", "-")
    )


def build_container_name(model_id: str) -> str:
    """Build the standard name for a vAquila-managed container."""
    return f"vaq-{_sanitize_model_id(model_id)}"


def _next_container_name(client: docker.DockerClient, model_id: str) -> str:
    """Return a unique container name for a model instance."""
    base_name = build_container_name(model_id)

    try:
        client.containers.get(base_name)
    except NotFound:
        return base_name

    suffix = 2
    while True:
        candidate = f"{base_name}-{suffix}"
        try:
            client.containers.get(candidate)
            suffix += 1
        except NotFound:
            return candidate


def _docker_client() -> docker.DockerClient:
    """Return a Docker client after validating daemon connectivity."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as exc:
        raise VaquilaError(
            "Docker daemon is unreachable. Start Docker Desktop (or the daemon) and try again."
        ) from exc


def check_docker_connection() -> None:
    """Validate that the Docker daemon connection is operational."""
    _docker_client()


def get_container(container_name: str) -> Container:
    """Return a container by name with domain-level error handling."""
    client = _docker_client()
    try:
        return client.containers.get(container_name)
    except NotFound as exc:
        raise VaquilaError(f"Container not found: {container_name}") from exc
    except DockerException as exc:
        raise VaquilaError(f"Unable to retrieve container {container_name}: {exc}") from exc


def _ensure_cache_dir(path: Path) -> Path:
    """Create the local Hugging Face cache directory if it does not exist."""
    raw = str(path)
    is_windows_host_abs = bool(re.match(r"^[A-Za-z]:[\\/]", raw))

    if is_windows_host_abs:
        # Path targets the host Docker daemon (Windows), not the vaq container.
        return path

    if not path.is_absolute():
        raise VaquilaError(
            "VAQ_HF_CACHE_HOST_PATH must be an absolute path readable by the Docker daemon. "
            "Windows example: C:/Users/<user>/GitRepository/vAquila/tmp/huggingface"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_model_container(
    model_id: str,
    host_port: int,
    gpu_index: int | None,
    gpu_utilization: float | None,
    cpu_utilization: float | None,
    max_num_seqs: int,
    max_model_len: int,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool,
    required_ratio: float | None,
    allow_long_context_override: bool,
    config: RuntimeConfig,
    quantization: str | None = None,
    kv_cache_dtype: str | None = None,
    compute_backend: str = "gpu",
) -> Container:
    """Create and start a vLLM container for a Hugging Face model."""
    client = _docker_client()

    name = _next_container_name(client, model_id)
    cache_path = _ensure_cache_dir(config.hf_cache_host_path)

    try:
        command = [
            model_id,
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-model-len",
            str(max_model_len),
        ]
        backend = compute_backend.lower().strip()
        if backend == "gpu":
            if gpu_utilization is None or gpu_index is None:
                raise VaquilaError("GPU launch requires gpu_index and gpu_utilization.")
            command.extend(["--gpu-memory-utilization", f"{gpu_utilization:.3f}"])
        elif backend == "cpu":
            # CPU image selects backend internally; some tags do not accept --device.
            pass
        else:
            raise VaquilaError(f"Unsupported compute backend: {compute_backend}")

        if tool_call_parser:
            command.extend(["--tool-call-parser", tool_call_parser])
        if reasoning_parser:
            command.extend(["--reasoning-parser", reasoning_parser])
        if quantization:
            command.extend(["--quantization", quantization])
        if kv_cache_dtype:
            command.extend(["--kv-cache-dtype", kv_cache_dtype])

        environment: dict[str, str] = {}
        if backend == "gpu" and gpu_index is not None:
            environment["NVIDIA_VISIBLE_DEVICES"] = str(gpu_index)
        if backend == "cpu":
            # Prevent vLLM from trying to infer a GPU device in CPU-only runs.
            environment["VLLM_TARGET_DEVICE"] = "cpu"
        if allow_long_context_override:
            environment["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        device_requests = None
        if backend == "gpu" and gpu_index is not None:
            device_requests = [DeviceRequest(device_ids=[str(gpu_index)], capabilities=[["gpu"]])]

        labels = {
            "com.vaquila.managed": "true",
            "com.vaquila.model_id": model_id,
            "com.vaquila.compute_backend": backend,
            "com.vaquila.gpu_index": "" if gpu_index is None else str(gpu_index),
            "com.vaquila.gpu_utilization": "" if gpu_utilization is None else f"{gpu_utilization:.3f}",
            "com.vaquila.cpu_utilization": "" if cpu_utilization is None else f"{cpu_utilization:.3f}",
            "com.vaquila.max_num_seqs": str(max_num_seqs),
            "com.vaquila.max_model_len": str(max_model_len),
            "com.vaquila.tool_call_parser": tool_call_parser or "",
            "com.vaquila.reasoning_parser": reasoning_parser or "",
            "com.vaquila.enable_thinking": "true" if enable_thinking else "false",
            "com.vaquila.required_ratio": "" if required_ratio is None else f"{required_ratio:.3f}",
            "com.vaquila.allow_long_context_override": "true" if allow_long_context_override else "false",
            "com.vaquila.quantization": quantization or "",
            "com.vaquila.kv_cache_dtype": kv_cache_dtype or "",
        }

        selected_image = config.cpu_image if backend == "cpu" else config.image

        nano_cpus: int | None = None
        if cpu_utilization is not None:
            cpu_count = os.cpu_count() or 1
            capped_ratio = max(0.0, min(1.0, cpu_utilization))
            nano_cpus = int(capped_ratio * cpu_count * 1_000_000_000)
            if nano_cpus <= 0:
                nano_cpus = None

        container = client.containers.run(
            image=selected_image,
            command=command,
            name=name,
            detach=True,
            ports={"8000/tcp": host_port},
            device_requests=device_requests,
            nano_cpus=nano_cpus,
            shm_size="2g",
            volumes={
                str(cache_path): {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw",
                }
            },
            environment=environment,
            labels=labels,
        )
        return container
    except DockerException as exc:
        raise VaquilaError(f"Unable to start vLLM container: {exc}") from exc


def list_managed_containers(snapshot_by_gpu: dict[int, GpuSnapshot] | None = None) -> list[ManagedContainer]:
    """List containers managed by vAquila."""
    client = _docker_client()
    containers = client.containers.list(all=True, filters={"label": "com.vaquila.managed=true"})
    rows: list[ManagedContainer] = []

    for container in containers:
        labels = container.labels or {}
        model_id = labels.get("com.vaquila.model_id", "unknown")
        instance_id = labels.get("com.vaquila.instance_id") or container.short_id
        compute_backend = labels.get("com.vaquila.compute_backend") or "gpu"
        gpu_idx_value = labels.get("com.vaquila.gpu_index")

        gpu_index: int | None = None
        gpu_used: int | None = None
        gpu_utilization: float | None = None
        max_num_seqs: int | None = None
        max_model_len: int | None = None
        tool_call_parser: str | None = None
        reasoning_parser: str | None = None
        enable_thinking: bool | None = None
        required_ratio: float | None = None
        allow_long_context_override: bool | None = None
        if gpu_idx_value is not None:
            try:
                gpu_index = int(gpu_idx_value)
            except ValueError:
                gpu_index = None

        gpu_utilization_value = labels.get("com.vaquila.gpu_utilization")
        if gpu_utilization_value is not None:
            try:
                gpu_utilization = float(gpu_utilization_value)
            except ValueError:
                gpu_utilization = None

        max_num_seqs_value = labels.get("com.vaquila.max_num_seqs")
        if max_num_seqs_value is not None:
            try:
                max_num_seqs = int(max_num_seqs_value)
            except ValueError:
                max_num_seqs = None

        max_model_len_value = labels.get("com.vaquila.max_model_len")
        if max_model_len_value is not None:
            try:
                max_model_len = int(max_model_len_value)
            except ValueError:
                max_model_len = None

        tool_call_parser_value = labels.get("com.vaquila.tool_call_parser")
        if tool_call_parser_value:
            tool_call_parser = tool_call_parser_value

        reasoning_parser_value = labels.get("com.vaquila.reasoning_parser")
        if reasoning_parser_value:
            reasoning_parser = reasoning_parser_value

        enable_thinking_value = labels.get("com.vaquila.enable_thinking")
        if enable_thinking_value is not None:
            enable_thinking = enable_thinking_value.lower() == "true"

        required_ratio_value = labels.get("com.vaquila.required_ratio")
        if required_ratio_value is not None:
            try:
                required_ratio = float(required_ratio_value)
            except ValueError:
                required_ratio = None

        allow_long_context_override_value = labels.get("com.vaquila.allow_long_context_override")
        if allow_long_context_override_value is not None:
            allow_long_context_override = allow_long_context_override_value.lower() == "true"

        if gpu_index is not None and snapshot_by_gpu and gpu_index in snapshot_by_gpu:
            snapshot = snapshot_by_gpu[gpu_index]
            if container.status == "running":
                if gpu_utilization is not None:
                    gpu_used = int(snapshot.total_bytes * gpu_utilization)
                else:
                    gpu_used = snapshot.used_bytes
            else:
                gpu_used = None

        host_port: int | None = None
        ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        mapping = ports.get("8000/tcp")
        if mapping and isinstance(mapping, list) and mapping[0].get("HostPort"):
            host_port = int(mapping[0]["HostPort"])

        rows.append(
            ManagedContainer(
                name=container.name,
                model_id=model_id,
                status=container.status,
                host_port=host_port,
                compute_backend=compute_backend,
                gpu_index=gpu_index,
                gpu_used_bytes=gpu_used,
                gpu_utilization=gpu_utilization,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                tool_call_parser=tool_call_parser,
                reasoning_parser=reasoning_parser,
                enable_thinking=enable_thinking,
                required_ratio=required_ratio,
                allow_long_context_override=allow_long_context_override,
                instance_id=instance_id,
            )
        )

    return rows


def stop_model_container(model_id: str) -> list[str]:
    """Stop and remove all container instances for a model ID."""
    client = _docker_client()
    containers = client.containers.list(
        all=True,
        filters={"label": ["com.vaquila.managed=true", f"com.vaquila.model_id={model_id}"]},
    )
    if not containers:
        candidate_name = build_container_name(model_id)
        try:
            maybe = client.containers.get(candidate_name)
            containers = [maybe]
        except NotFound:
            raise VaquilaError(f"No container found for model `{model_id}`.")

    removed_names: list[str] = []
    for container in containers:
        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove(v=True)
            removed_names.append(container.name)
        except DockerException as exc:
            raise VaquilaError(f"Failed to remove container {container.name}: {exc}") from exc

    return removed_names


def stop_containers_by_name(container_names: list[str]) -> list[str]:
    """Stop and remove a list of containers by name."""
    client = _docker_client()
    removed_names: list[str] = []

    for container_name in container_names:
        try:
            container = client.containers.get(container_name)
        except NotFound:
            continue
        except DockerException as exc:
            raise VaquilaError(f"Unable to retrieve container {container_name}: {exc}") from exc

        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove(v=True)
            removed_names.append(container.name)
        except DockerException as exc:
            raise VaquilaError(f"Failed to remove container {container.name}: {exc}") from exc

    return removed_names
