"""vLLM startup helpers and log parsing utilities."""

from __future__ import annotations

import re
import time
from contextlib import suppress

from docker.errors import DockerException, NotFound
from rich.console import Console

from vaquila.docker_service import get_container
from vaquila.exceptions import VaquilaError

_READY_MARKERS = (
    "Application startup complete",
    "Uvicorn running on",
)

_DOWNLOAD_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("snapshot_download", "Downloading Hugging Face weights..."),
    ("hf_hub_download", "Downloading model files..."),
    ("download_weights_from_hf", "Downloading model weights..."),
    ("file_download", "Downloading artifacts..."),
)

_LOAD_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Starting to load model", "Loading model into VRAM..."),
    ("Resolved architecture", "Model architecture detected..."),
    ("Initializing a V1 LLM engine", "Initializing vLLM engine..."),
)

_HF_PROGRESS_RE = re.compile(r"(\d{1,3})%\s+Completed\s+\|\s*(\d+)/(\d+)")
_KV_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([0-9]+(?:\.[0-9]+)?)x"
)


def _render_progress_bar(percent: int, width: int = 24) -> str:
    """Render a compact ASCII progress bar from a 0-100 percentage."""
    clamped = max(0, min(100, percent))
    filled = int(round((clamped / 100.0) * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _extract_hf_progress(log_text: str) -> tuple[int, int, int] | None:
    """Extract latest Hugging Face shard download progress from startup logs."""
    matches = _HF_PROGRESS_RE.findall(log_text)
    if not matches:
        return None

    percent_text, current_text, total_text = matches[-1]
    with suppress(ValueError):
        percent = int(percent_text)
        current = int(current_text)
        total = int(total_text)
        return percent, current, total

    return None


def clean_log_line(line: str) -> str:
    """Clean a vLLM log prefix for CLI display."""
    cleaned = re.sub(r"^\([^)]*\)\s*", "", line).strip()
    return cleaned


def extract_startup_hint(log_text: str) -> str:
    """Return a readable startup progress hint from vLLM logs."""
    progress_matches = _HF_PROGRESS_RE.findall(log_text)
    if progress_matches:
        percent, current, total = progress_matches[-1]
        return f"Loading weights (shards): {percent}% ({current}/{total})"

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
            return f"Error detected: {cleaned[:140]}"
        if any(token in cleaned for token in ("INFO", "WARNING", "Starting", "loading", "download")):
            return cleaned[:140]

    return "Starting vLLM server..."


def extract_root_error(log_text: str) -> str | None:
    """Extract a useful root cause from startup logs."""
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
    """Extract observed KV max concurrency for a given context size."""
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
    """Follow vLLM startup logs and wait until the API is ready."""
    started = time.monotonic()
    last_hint = "Starting vLLM server..."
    last_log_text = ""
    last_reported_hf_percent = -1
    last_download_phase_hint = ""

    with console.status(f"[cyan]{last_hint}[/cyan]", spinner="dots") as status:
        while time.monotonic() - started < timeout_seconds:
            elapsed = int(time.monotonic() - started)
            try:
                runtime_container = get_container(container_name)
                runtime_container.reload()
                log_text = runtime_container.logs(tail=250).decode("utf-8", errors="replace")
                last_log_text = log_text
            except VaquilaError as exc:
                root_error = extract_root_error(last_log_text)
                detail = f" Last known root cause: {root_error}" if root_error else ""
                raise VaquilaError(
                    f"Container `{container_name}` disappeared during initialization.{detail}"
                ) from exc
            except (NotFound, DockerException) as exc:
                root_error = extract_root_error(last_log_text)
                detail = f" Last known root cause: {root_error}" if root_error else ""
                raise VaquilaError(
                    f"Container `{container_name}` became unavailable during initialization.{detail}"
                ) from exc

            if not log_text.strip():
                hint = (
                    "Preparing vLLM runtime... "
                    f"({elapsed}s, first launch may trigger Hugging Face downloads)"
                )
            else:
                hint = extract_startup_hint(log_text)

            if hint != last_hint:
                last_hint = hint
                status.update(f"[cyan]{hint}[/cyan]")

            hf_progress = _extract_hf_progress(log_text)
            if hf_progress is not None:
                percent, current, total = hf_progress
                if percent >= 100 or percent - last_reported_hf_percent >= 2:
                    bar = _render_progress_bar(percent)
                    console.print(
                        f"[startup] Hugging Face download {bar} {percent}% ({current}/{total} shards)"
                    )
                    last_reported_hf_percent = percent
            elif "download" in hint.lower() and hint != last_download_phase_hint:
                console.print(f"[startup] {hint}")
                last_download_phase_hint = hint

            if any(marker in log_text for marker in _READY_MARKERS):
                return

            if runtime_container.status in {"exited", "dead"}:
                root_error = extract_root_error(log_text)
                detail = (
                    f" Probable cause: {root_error}"
                    if root_error
                    else " Check container logs for detailed root cause."
                )
                raise VaquilaError(
                    f"Container `{container_name}` stopped during initialization.{detail}"
                )

            time.sleep(2)

    raise VaquilaError(
        f"Startup timeout ({timeout_seconds}s). The model may still be downloading. "
        f"Follow logs: docker logs -f {container_name}"
    )
