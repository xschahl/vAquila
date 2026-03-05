"""Services Docker pour l'orchestration des conteneurs vLLM."""

from __future__ import annotations

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
    """Sanitize un model id pour générer un nom de conteneur lisible."""
    return (
        model_id.lower()
        .replace("/", "-")
        .replace("_", "-")
        .replace(".", "-")
        .replace(":", "-")
    )


def build_container_name(model_id: str) -> str:
    """Construit le nom standard d'un conteneur managé par vAquila."""
    return f"vaq-{_sanitize_model_id(model_id)}"


def _next_container_name(client: docker.DockerClient, model_id: str) -> str:
    """Retourne un nom de conteneur unique pour une instance de modèle."""
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
    """Retourne un client Docker en validant la connexion démon."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except DockerException as exc:
        raise VaquilaError(
            "Daemon Docker inaccessible. Démarre Docker Desktop (ou le daemon) puis réessaie."
        ) from exc


def check_docker_connection() -> None:
    """Valide que la connexion au daemon Docker est opérationnelle."""
    _docker_client()


def get_container(container_name: str) -> Container:
    """Retourne un conteneur par son nom avec gestion d'erreur métier."""
    client = _docker_client()
    try:
        return client.containers.get(container_name)
    except NotFound as exc:
        raise VaquilaError(f"Conteneur introuvable: {container_name}") from exc
    except DockerException as exc:
        raise VaquilaError(f"Impossible de récupérer le conteneur {container_name}: {exc}") from exc


def _ensure_cache_dir(path: Path) -> Path:
    """Crée le dossier de cache Hugging Face local s'il n'existe pas."""
    raw = str(path)
    is_windows_host_abs = bool(re.match(r"^[A-Za-z]:[\\/]", raw))

    if is_windows_host_abs:
        # Le chemin est destiné au daemon Docker hôte (Windows), pas au conteneur vaq.
        return path

    if not path.is_absolute():
        raise VaquilaError(
            "VAQ_HF_CACHE_HOST_PATH doit être un chemin absolu lisible par Docker daemon. "
            "Exemple Windows: C:/Users/<user>/GitRepository/vAquila/tmp/huggingface"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_model_container(
    model_id: str,
    host_port: int,
    gpu_index: int,
    gpu_utilization: float,
    max_num_seqs: int,
    max_model_len: int,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool,
    required_ratio: float,
    allow_long_context_override: bool,
    config: RuntimeConfig,
) -> Container:
    """Crée et démarre un conteneur vLLM pour un modèle HF."""
    client = _docker_client()

    name = _next_container_name(client, model_id)
    cache_path = _ensure_cache_dir(config.hf_cache_host_path)

    try:
        command = [
            "--model",
            model_id,
            "--gpu-memory-utilization",
            f"{gpu_utilization:.3f}",
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-model-len",
            str(max_model_len),
        ]
        if tool_call_parser:
            command.extend(["--tool-call-parser", tool_call_parser])
        if reasoning_parser:
            command.extend(["--reasoning-parser", reasoning_parser])

        environment = {"NVIDIA_VISIBLE_DEVICES": str(gpu_index)}
        if allow_long_context_override:
            environment["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        container = client.containers.run(
            image=config.image,
            command=command,
            name=name,
            detach=True,
            ports={"8000/tcp": host_port},
            device_requests=[
                DeviceRequest(device_ids=[str(gpu_index)], capabilities=[["gpu"]])
            ],
            shm_size="2g",
            volumes={
                str(cache_path): {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw",
                }
            },
            environment=environment,
            labels={
                "com.vaquila.managed": "true",
                "com.vaquila.model_id": model_id,
                "com.vaquila.gpu_index": str(gpu_index),
                "com.vaquila.gpu_utilization": f"{gpu_utilization:.3f}",
                "com.vaquila.max_num_seqs": str(max_num_seqs),
                "com.vaquila.max_model_len": str(max_model_len),
                "com.vaquila.tool_call_parser": tool_call_parser or "",
                "com.vaquila.reasoning_parser": reasoning_parser or "",
                "com.vaquila.enable_thinking": "true" if enable_thinking else "false",
                "com.vaquila.required_ratio": f"{required_ratio:.3f}",
                "com.vaquila.allow_long_context_override": "true" if allow_long_context_override else "false",
            },
        )
        return container
    except DockerException as exc:
        raise VaquilaError(f"Impossible de lancer le conteneur vLLM: {exc}") from exc


def list_managed_containers(snapshot_by_gpu: dict[int, GpuSnapshot] | None = None) -> list[ManagedContainer]:
    """Liste les conteneurs gérés par vAquila."""
    client = _docker_client()
    containers = client.containers.list(all=True, filters={"label": "com.vaquila.managed=true"})
    rows: list[ManagedContainer] = []

    for container in containers:
        labels = container.labels or {}
        model_id = labels.get("com.vaquila.model_id", "unknown")
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
            )
        )

    return rows


def stop_model_container(model_id: str) -> list[str]:
    """Stoppe et supprime toutes les instances de conteneur d'un model id."""
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
            raise VaquilaError(f"Aucun conteneur trouvé pour le modèle `{model_id}`.")

    removed_names: list[str] = []
    for container in containers:
        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove(v=True)
            removed_names.append(container.name)
        except DockerException as exc:
            raise VaquilaError(f"Échec de suppression du conteneur {container.name}: {exc}") from exc

    return removed_names


def stop_containers_by_name(container_names: list[str]) -> list[str]:
    """Stoppe et supprime une liste de conteneurs par nom."""
    client = _docker_client()
    removed_names: list[str] = []

    for container_name in container_names:
        try:
            container = client.containers.get(container_name)
        except NotFound:
            continue
        except DockerException as exc:
            raise VaquilaError(f"Impossible de récupérer le conteneur {container_name}: {exc}") from exc

        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove(v=True)
            removed_names.append(container.name)
        except DockerException as exc:
            raise VaquilaError(f"Échec de suppression du conteneur {container.name}: {exc}") from exc

    return removed_names
