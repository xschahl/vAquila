"""Helpers cache Hugging Face et résolution de limites de contexte."""

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
    """Affiche des octets en Gio lisibles."""
    if value_bytes is None:
        return "n/a"
    return f"{value_bytes / (1024**3):.2f} Gio"


def check_hf_cache_path() -> str:
    """Vérifie que le cache Hugging Face est accessible en lecture/écriture."""
    path = CONFIG.hf_cache_host_path
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise VaquilaError(f"Le chemin de cache n'est pas un dossier: {path}")

    readable = os.access(path, os.R_OK)
    writable = os.access(path, os.W_OK)
    if not readable or not writable:
        raise VaquilaError(f"Permissions insuffisantes sur le cache: {path} (read={readable}, write={writable})")

    return str(path)


def model_cache_repo_dir(model_id: str) -> str:
    """Construit le nom de dossier repo Hugging Face dans le cache local."""
    normalized = model_id.strip().replace("/", "--")
    return f"models--{normalized}"


def extract_model_context_limit(config_payload: dict[str, object]) -> int | None:
    """Extrait la limite de contexte depuis un config.json HF."""
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
    """Retourne la racine du cache hub Hugging Face lisible depuis le runtime CLI."""
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
    """Lit le config.json depuis le cache local HF si disponible."""
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
    """Récupère config.json depuis Hugging Face Hub en fallback réseau."""
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
    """Résout la limite de contexte modèle depuis cache HF puis réseau."""
    cached_payload = read_cached_model_config(model_id)
    if cached_payload is not None:
        cached_limit = extract_model_context_limit(cached_payload)
        if cached_limit is not None:
            return cached_limit

    remote_payload = fetch_remote_model_config(model_id)
    if remote_payload is not None:
        return extract_model_context_limit(remote_payload)

    return None


def cache_dir_to_model_id(cache_dir: Path) -> str:
    """Convertit un dossier cache HF en model_id lisible."""
    name = cache_dir.name
    if not name.startswith("models--"):
        return name
    return name.replace("models--", "", 1).replace("--", "/")


def dir_size_bytes(path: Path) -> int:
    """Calcule la taille totale d'un dossier."""
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            with suppress(OSError):
                total += file_path.stat().st_size
    return total


def list_cached_model_dirs() -> list[Path]:
    """Liste les dossiers de modèles présents dans le cache local."""
    root = hub_cache_root()
    if not root.exists():
        return []
    return sorted([entry for entry in root.glob("models--*") if entry.is_dir()], key=lambda p: p.name)


def purge_model_cache(model_id: str) -> bool:
    """Supprime le cache local d'un modèle Hugging Face si présent."""
    cache_root = hub_cache_root()
    target = cache_root / model_cache_repo_dir(model_id)

    if not target.exists():
        return False

    shutil.rmtree(target)
    return True
