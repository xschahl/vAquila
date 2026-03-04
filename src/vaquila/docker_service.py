"""Services Docker pour l'orchestration des conteneurs vLLM."""

from __future__ import annotations

from pathlib import Path

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


def _ensure_cache_dir(path: Path) -> Path:
    """Crée le dossier de cache Hugging Face local s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_model_container(
    model_id: str,
    host_port: int,
    gpu_index: int,
    gpu_utilization: float,
    config: RuntimeConfig,
) -> Container:
    """Crée et démarre un conteneur vLLM pour un modèle HF."""
    client = _docker_client()

    existing = client.containers.list(
        all=True,
        filters={"label": ["com.vaquila.managed=true", f"com.vaquila.model_id={model_id}"]},
    )
    if existing:
        raise VaquilaError(
            f"Un conteneur existe déjà pour ce modèle: {existing[0].name}. Utilise `vaq stop {model_id}`."
        )

    name = build_container_name(model_id)
    cache_path = _ensure_cache_dir(config.hf_cache_host_path)

    try:
        container = client.containers.run(
            image=config.image,
            command=[
                "--model",
                model_id,
                "--gpu-memory-utilization",
                f"{gpu_utilization:.3f}",
            ],
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
            environment={"NVIDIA_VISIBLE_DEVICES": str(gpu_index)},
            labels={
                "com.vaquila.managed": "true",
                "com.vaquila.model_id": model_id,
                "com.vaquila.gpu_index": str(gpu_index),
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
        if gpu_idx_value is not None:
            try:
                gpu_index = int(gpu_idx_value)
            except ValueError:
                gpu_index = None

        if gpu_index is not None and snapshot_by_gpu and gpu_index in snapshot_by_gpu:
            gpu_used = snapshot_by_gpu[gpu_index].used_bytes

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
            )
        )

    return rows


def stop_model_container(model_id: str) -> str:
    """Stoppe et supprime le conteneur associé à un model id."""
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

    container = containers[0]
    try:
        if container.status == "running":
            container.stop(timeout=10)
        container.remove(v=True)
        return container.name
    except DockerException as exc:
        raise VaquilaError(f"Échec de suppression du conteneur {container.name}: {exc}") from exc
