"""Hugging Face cache helpers and context-limit resolution."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import suppress
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from vaquila.config import CONFIG
from vaquila.exceptions import VaquilaError


def format_gb(value_bytes: int | None) -> str:
    """Render bytes as a readable GiB value."""
    if value_bytes is None:
        return "n/a"
    return f"{value_bytes / (1024**3):.2f} GiB"


def check_hf_cache_path() -> str:
    """Check that the Hugging Face cache is readable and writable."""
    path = CONFIG.hf_cache_host_path
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise VaquilaError(f"Cache path is not a directory: {path}")

    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    if not readable or not writable:
        raise VaquilaError(f"Insufficient cache permissions: {path} (read={readable}, write={writable})")

    return str(path)


def model_cache_repo_dir(model_id: str) -> str:
    """Build the Hugging Face repo directory name in local cache."""
    normalized = model_id.strip().replace("/", "--")
    return f"models--{normalized}"


def extract_model_context_limit(config_payload: dict[str, object]) -> int | None:
    """Extract context limit from a Hugging Face `config.json`."""
    candidate_keys = (
        "max_position_embeddings",
        "model_max_length",
        "n_positions",
        "max_seq_len",
        "seq_length",
    )
    for key in candidate_keys:
        value = config_payload.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def hub_cache_root() -> Path:
    """Return the Hugging Face hub cache root readable by the CLI runtime."""
    configured = CONFIG.hf_cache_host_path / "hub"
    mounted = Path("/root/.cache/huggingface/hub")

    configured_has_models = configured.exists() and any(configured.glob("models--*"))
    mounted_has_models = mounted.exists() and any(mounted.glob("models--*"))

    if configured_has_models:
        return configured
    if mounted_has_models:
        return mounted

    if configured.exists():
        return configured
    if mounted.exists():
        return mounted

    return configured


def read_cached_model_config(model_id: str) -> dict[str, object] | None:
    """Read `config.json` from local Hugging Face cache when available."""
    repo_dir = hub_cache_root() / model_cache_repo_dir(model_id)
    if not repo_dir.exists():
        return None

    refs_main = repo_dir / "refs" / "main"
    snapshot_id: str | None = None
    if refs_main.exists():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip() or None

    candidate_paths: list[Path] = []
    if snapshot_id:
        candidate_paths.append(repo_dir / "snapshots" / snapshot_id / "config.json")
    candidate_paths.extend(repo_dir.glob("snapshots/*/config.json"))

    for config_path in candidate_paths:
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload

    return None


def fetch_remote_model_config(model_id: str) -> dict[str, object] | None:
    """Fetch `config.json` from Hugging Face Hub as a network fallback."""
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        with urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
    except (URLError, TimeoutError, ValueError):
        return None

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def resolve_model_context_limit(model_id: str) -> int | None:
    """Resolve model context limit from HF cache first, then network."""
    cached_payload = read_cached_model_config(model_id)
    if cached_payload is not None:
        cached_limit = extract_model_context_limit(cached_payload)
        if cached_limit is not None:
            return cached_limit

    remote_payload = fetch_remote_model_config(model_id)
    if remote_payload is not None:
        return extract_model_context_limit(remote_payload)

    return None


def resolve_model_config(model_id: str) -> dict[str, object] | None:
    """Resolve model `config.json` from local cache, then Hugging Face Hub."""
    cached_payload = read_cached_model_config(model_id)
    if cached_payload is not None:
        return cached_payload
    return fetch_remote_model_config(model_id)


def cache_dir_to_model_id(cache_dir: Path) -> str:
    """Convert a Hugging Face cache directory to a readable model ID."""
    name = cache_dir.name
    if not name.startswith("models--"):
        return name
    return name.replace("models--", "", 1).replace("--", "/")


def dir_size_bytes(path: Path) -> int:
    """Compute total directory size in bytes."""
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            with suppress(OSError):
                total += file_path.stat().st_size
    return total


def list_cached_model_dirs() -> list[Path]:
    """List model directories present in local cache."""
    root = hub_cache_root()
    if not root.exists():
        return []
    return sorted([entry for entry in root.glob("models--*") if entry.is_dir()], key=lambda p: p.name)


def purge_model_cache(model_id: str) -> bool:
    """Remove local cache for a Hugging Face model if present."""
    cache_root = hub_cache_root()
    target = cache_root / model_cache_repo_dir(model_id)

    if not target.exists():
        return False

    shutil.rmtree(target)
    return True
