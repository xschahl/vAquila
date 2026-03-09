"""vAquila runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings for launching vLLM containers."""

    image: str = "vllm/vllm-openai:latest"
    cpu_image: str = "vllm/vllm-openai:latest"
    default_host_port: int = 8000
    security_vram_buffer_gb: float = 1.5
    hf_cache_host_path: Path = Path.home() / ".cache" / "huggingface"
    inference_base_url: str = "http://localhost:8000"


def _default_inference_base_url() -> str:
    """Return the default API URL based on execution context."""
    if Path("/.dockerenv").exists():
        return "http://host.docker.internal:8000"
    return "http://localhost:8000"


def load_config() -> RuntimeConfig:
    """Load runtime configuration from environment variables."""
    image = os.getenv("VAQ_VLLM_IMAGE", "vllm/vllm-openai:latest")
    cpu_image = os.getenv("VAQ_VLLM_CPU_IMAGE", image)
    default_host_port = int(os.getenv("VAQ_DEFAULT_HOST_PORT", "8000"))
    security_vram_buffer_gb = float(os.getenv("VAQ_SECURITY_VRAM_BUFFER_GB", "1.5"))
    hf_cache_host_path = Path(
        os.getenv("VAQ_HF_CACHE_HOST_PATH", str(Path.home() / ".cache" / "huggingface"))
    )
    inference_base_url = os.getenv("VAQ_INFERENCE_BASE_URL", _default_inference_base_url())

    return RuntimeConfig(
        image=image,
        cpu_image=cpu_image,
        default_host_port=default_host_port,
        security_vram_buffer_gb=security_vram_buffer_gb,
        hf_cache_host_path=hf_cache_host_path,
        inference_base_url=inference_base_url,
    )


CONFIG = load_config()
