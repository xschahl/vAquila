"""Helpers startup vLLM et parsing de logs."""

from __future__ import annotations

import re
import time
from contextlib import suppress

from rich.console import Console

from vaquila.docker_service import get_container
from vaquila.exceptions import VaquilaError

_READY_MARKERS = (
    "Application startup complete",
    "Uvicorn running on",
)

_DOWNLOAD_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("snapshot_download", "Téléchargement des poids Hugging Face..."),
    ("hf_hub_download", "Téléchargement des fichiers modèle..."),
    ("download_weights_from_hf", "Téléchargement des poids modèle..."),
    ("file_download", "Téléchargement des artefacts..."),
)

_LOAD_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Starting to load model", "Chargement du modèle en VRAM..."),
    ("Resolved architecture", "Architecture modèle détectée..."),
    ("Initializing a V1 LLM engine", "Initialisation du moteur vLLM..."),
)

_HF_PROGRESS_RE = re.compile(r"(\d{1,3})%\s+Completed\s+\|\s*(\d+)/(\d+)")
_KV_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([0-9]+(?:\.[0-9]+)?)x"
)


def clean_log_line(line: str) -> str:
    """Nettoie un préfixe de log vLLM pour affichage CLI."""
    cleaned = re.sub(r"^\([^)]*\)\s*", "", line).strip()
    return cleaned


def extract_startup_hint(log_text: str) -> str:
    """Retourne un message de progression lisible à partir des logs vLLM."""
    progress_matches = _HF_PROGRESS_RE.findall(log_text)
    if progress_matches:
        percent, current, total = progress_matches[-1]
        return f"Chargement des poids (shards): {percent}% ({current}/{total})"

    for marker, message in _DOWNLOAD_PHASE_MARKERS:
        if marker in log_text:
            return message

    for marker, message in _LOAD_PHASE_MARKERS:
        if marker in log_text:
            return message

    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    for line in reversed(lines):
        cleaned = clean_log_line(line)
        if "ERROR" in cleaned:
            return f"Erreur détectée: {cleaned[:140]}"
        if any(token in cleaned for token in ("INFO", "WARNING", "Starting", "loading", "download")):
            return cleaned[:140]

    return "Démarrage du serveur vLLM..."


def extract_root_error(log_text: str) -> str | None:
    """Extrait une cause racine utile depuis les logs de startup."""
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]

    disk_error_tokens = (
        "Not enough free disk space",
        "No space left on device",
    )
    for line in lines:
        if any(token in line for token in disk_error_tokens):
            return clean_log_line(line)

    value_error_lines = [line for line in lines if "ValueError:" in line]
    if value_error_lines:
        return clean_log_line(value_error_lines[-1])

    specific_runtime_lines = [
        line for line in lines if "RuntimeError:" in line and "Engine core initialization failed" not in line
    ]
    if specific_runtime_lines:
        return clean_log_line(specific_runtime_lines[-1])

    runtime_error_lines = [line for line in lines if "RuntimeError:" in line]
    if runtime_error_lines:
        return clean_log_line(runtime_error_lines[-1])

    error_lines = [line for line in lines if "ERROR" in line]
    if error_lines:
        return clean_log_line(error_lines[-1])

    return None


def extract_kv_max_concurrency(log_text: str, request_tokens: int) -> float | None:
    """Extrait la concurrence max KV observée pour un contexte donné."""
    matches = _KV_CONCURRENCY_RE.findall(log_text)
    if not matches:
        return None

    target_value: float | None = None
    fallback_value: float | None = None
    for tokens_text, concurrency_text in matches:
        with suppress(ValueError):
            parsed_tokens = int(tokens_text.replace(",", ""))
            parsed_concurrency = float(concurrency_text)
            fallback_value = parsed_concurrency
            if parsed_tokens == request_tokens:
                target_value = parsed_concurrency

    return target_value if target_value is not None else fallback_value


def wait_until_model_ready(console: Console, container_name: str, timeout_seconds: int = 900) -> None:
    """Suit les logs de startup vLLM et attend que l'API soit prête."""
    started = time.monotonic()
    last_hint = "Démarrage du serveur vLLM..."

    with console.status(f"[cyan]{last_hint}[/cyan]", spinner="dots") as status:
        while time.monotonic() - started < timeout_seconds:
            elapsed = int(time.monotonic() - started)
            runtime_container = get_container(container_name)
            runtime_container.reload()
            log_text = runtime_container.logs(tail=250).decode("utf-8", errors="replace")

            if not log_text.strip():
                hint = (
                    "Préparation du runtime vLLM... "
                    f"({elapsed}s, premier lancement = téléchargement Hugging Face possible)"
                )
            else:
                hint = extract_startup_hint(log_text)

            if hint != last_hint:
                last_hint = hint
                status.update(f"[cyan]{hint}[/cyan]")

            if any(marker in log_text for marker in _READY_MARKERS):
                return

            if runtime_container.status in {"exited", "dead"}:
                root_error = extract_root_error(log_text)
                detail = (
                    f" Cause probable: {root_error}"
                    if root_error
                    else " Consulte les logs du conteneur pour la cause détaillée."
                )
                raise VaquilaError(
                    f"Le conteneur `{container_name}` s'est arrêté pendant l'initialisation.{detail}"
                )

            time.sleep(2)

    raise VaquilaError(
        f"Timeout de démarrage ({timeout_seconds}s). Le modèle est peut-être encore en téléchargement. "
        f"Suis les logs: docker logs -f {container_name}"
    )
