"""Configuration runtime de vAquila."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Paramètres de runtime pour le lancement de conteneurs vLLM."""

    image: str = "vllm/vllm-openai:latest"
    default_host_port: int = 8000
    security_vram_buffer_gb: float = 1.5
    hf_cache_host_path: Path = Path.home() / ".cache" / "huggingface"
    inference_base_url: str = "http://localhost:8000"


def _default_inference_base_url() -> str:
    """Retourne l'URL API par défaut selon le contexte d'exécution."""
    if Path("/.dockerenv").exists():
        return "http://host.docker.internal:8000"
    return "http://localhost:8000"


def load_config() -> RuntimeConfig:
    """Charge la configuration runtime depuis l'environnement."""
    image = os.getenv("VAQ_VLLM_IMAGE", "vllm/vllm-openai:latest")
    default_host_port = int(os.getenv("VAQ_DEFAULT_HOST_PORT", "8000"))
    security_vram_buffer_gb = float(os.getenv("VAQ_SECURITY_VRAM_BUFFER_GB", "1.5"))
    hf_cache_host_path = Path(
        os.getenv("VAQ_HF_CACHE_HOST_PATH", str(Path.home() / ".cache" / "huggingface"))
    )
    inference_base_url = os.getenv("VAQ_INFERENCE_BASE_URL", _default_inference_base_url())

    return RuntimeConfig(
        image=image,
        default_host_port=default_host_port,
        security_vram_buffer_gb=security_vram_buffer_gb,
        hf_cache_host_path=hf_cache_host_path,
        inference_base_url=inference_base_url,
    )


CONFIG = load_config()
