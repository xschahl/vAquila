"""Local Web UI services for vAquila."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from time import sleep
from uuid import uuid4

import typer
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rich.console import Console

from vaquila.cli_helpers import (
    cache_dir_to_model_id,
    check_hf_cache_path,
    dir_size_bytes,
    list_cached_model_dirs,
    purge_model_cache,
)
from vaquila.commands import run as run_command_module
from vaquila.commands.run import cmd_run
from vaquila.config import CONFIG
from vaquila.docker_service import get_container, list_managed_containers, stop_model_container
from vaquila.exceptions import VaquilaError
from vaquila.gpu import read_all_gpu_snapshots, read_gpu_snapshot
from vaquila.inference import run_inference


@dataclass
class RunTask:
    """Track asynchronous model launch tasks from the Web UI."""

    id: str
    model_id: str
    status: str
    message: str
    container_name: str | None
    started_at: str
    finished_at: str | None
    events: list[dict[str, str]] = field(default_factory=list)


class RunRequest(BaseModel):
    """Run request payload for launching a model container."""

    model_id: str = Field(min_length=1)
    port: int = Field(default=CONFIG.default_host_port, ge=1, le=65535)
    gpu_index: int = Field(default=0, ge=0)
    buffer_gb: float | None = Field(default=None, gt=0)
    startup_timeout: int = Field(default=900, ge=10)
    max_num_seqs: int = Field(default=1, ge=1)
    max_model_len: int = Field(default=16384, ge=1)
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    enable_thinking: bool = True
    allow_long_context_override: bool = False
    quantization: str = "auto"
    kv_cache_dtype: str = "auto"


class StopRequest(BaseModel):
    """Stop request payload."""

    model_id: str = Field(min_length=1)
    purge_cache: bool = False


class RemoveRequest(BaseModel):
    """Cache removal payload."""

    model_id: str = Field(min_length=1)


class InferRequest(BaseModel):
    """Inference request payload."""

    model_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    base_url: str = Field(default=CONFIG.inference_base_url, min_length=1)
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.2)
    timeout: int = Field(default=120, ge=1)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_optional_text(value: str | None) -> str:
    """Normalize optional parser values so run command stays non-interactive."""
    if value is None:
        return ""
    return value.strip()


def _remove_model_cache_or_raise(model_id: str) -> bool:
    """Remove one cached model after ensuring no matching container is still running."""
    try:
        managed_containers = list_managed_containers()
    except VaquilaError:
        managed_containers = []

    running_for_model = [
        container
        for container in managed_containers
        if container.model_id == model_id and container.status == "running"
    ]
    if running_for_model:
        names = ", ".join(container.name for container in running_for_model)
        raise HTTPException(
            status_code=400,
            detail=f"Model `{model_id}` is still running ({names}). Stop it first.",
        )

    return purge_model_cache(model_id)


def create_web_app() -> FastAPI:
    """Create and configure the FastAPI Web UI application."""
    app = FastAPI(title="vAquila Web UI", version="0.1.0")

    assets_dir = Path(__file__).resolve().parent / "assets"
    ui_dir = Path(__file__).resolve().parent / "webui_static"

    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    if ui_dir.exists():
        app.mount("/webui", StaticFiles(directory=str(ui_dir)), name="webui")

    index_path = ui_dir / "index.html"

    tasks: dict[str, RunTask] = {}
    task_log_lines: dict[str, list[str]] = {}
    tasks_lock = Lock()

    def _append_task_log_line(task_id: str, line: str) -> None:
        normalized = line.strip()
        if not normalized:
            return

        with tasks_lock:
            lines = task_log_lines.get(task_id)
            if lines is None:
                return
            lines.append(normalized)
            if len(lines) > 4000:
                del lines[:-2500]

    def _append_task_event(task_id: str, message: str, status: str | None = None) -> None:
        with tasks_lock:
            current = tasks.get(task_id)
            if current is None:
                return
            if status is not None:
                current.status = status
            current.message = message
            current.events.append(
                {
                    "timestamp": _utc_now(),
                    "status": current.status,
                    "message": message,
                }
            )
            lines = task_log_lines.get(task_id)
            if lines is not None:
                lines.append(f"[{current.events[-1]['timestamp']}] [{current.status}] {message}")
                if len(lines) > 4000:
                    del lines[:-2500]

    def _append_container_logs_to_task(task_id: str, container_name: str | None, tail: int = 240) -> None:
        if not container_name:
            return

        try:
            container = get_container(container_name)
            logs_text = container.logs(tail=max(20, min(tail, 1500))).decode(
                "utf-8",
                errors="replace",
            )
        except VaquilaError as exc:
            _append_task_log_line(
                task_id,
                f"[container] Unable to retrieve logs for {container_name}: {exc}",
            )
            return

        _append_task_log_line(task_id, f"[container] ---- docker logs: {container_name} ----")
        for line in logs_text.splitlines():
            if line.strip():
                _append_task_log_line(task_id, f"[container] {line}")
        _append_task_log_line(task_id, f"[container] ---- end docker logs: {container_name} ----")

    class _TaskLogStream:
        """Stream stdout/stderr content into one task log buffer."""

        class _BinaryBuffer:
            """Accept binary writes and forward them to the text stream."""

            def __init__(self, parent: "_TaskLogStream") -> None:
                self._parent = parent

            def write(self, data: bytes | bytearray) -> int:
                if not data:
                    return 0
                text = bytes(data).decode("utf-8", errors="replace")
                return self._parent.write(text)

            def flush(self) -> None:
                self._parent.flush()

        def __init__(self, task_id: str, channel: str) -> None:
            self.task_id = task_id
            self.channel = channel
            self._buffer = ""
            self.buffer = self._BinaryBuffer(self)
            self.encoding = "utf-8"
            self.errors = "replace"

        def write(self, data: str | bytes) -> int:
            if not data:
                return 0

            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")

            self._buffer += data.replace("\r\n", "\n").replace("\r", "\n")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    _append_task_log_line(self.task_id, f"[{self.channel}] {line}")
            return len(data)

        def flush(self) -> None:
            if self._buffer.strip():
                _append_task_log_line(self.task_id, f"[{self.channel}] {self._buffer}")
            self._buffer = ""

        def isatty(self) -> bool:
            return False

    def _set_task(task_id: str, status: str, message: str, finished: bool = False) -> None:
        _append_task_event(task_id, message, status=status)
        if finished:
            with tasks_lock:
                current = tasks.get(task_id)
                if current is None:
                    return
                current.finished_at = _utc_now()

    def _set_task_container(task_id: str, container_name: str | None) -> None:
        with tasks_lock:
            current = tasks.get(task_id)
            if current is None:
                return
            current.container_name = container_name

    def _launch_task(task_id: str, payload: RunRequest) -> None:
        _set_task(task_id, "running", "Launching model container...")
        known_container_name: str | None = None
        selected_container_name: str | None = None
        try:
            try:
                before_rows = list_managed_containers()
                before_model_names = {
                    row.name
                    for row in before_rows
                    if row.model_id == payload.model_id
                }
            except VaquilaError:
                before_model_names = set()

            runner_state: dict[str, object] = {"error": None, "exit_code": None}

            def _run_command() -> None:
                stdout_stream = _TaskLogStream(task_id, "stdout")
                stderr_stream = _TaskLogStream(task_id, "stderr")
                previous_console = run_command_module.console
                try:
                    run_command_module.console = Console(
                        file=stdout_stream,
                        force_terminal=False,
                        color_system=None,
                    )
                    with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                        cmd_run(
                            model_id=payload.model_id,
                            port=payload.port,
                            gpu_index=payload.gpu_index,
                            buffer_gb=payload.buffer_gb,
                            startup_timeout=payload.startup_timeout,
                            max_num_seqs=payload.max_num_seqs,
                            max_model_len=payload.max_model_len,
                            tool_call_parser=_normalize_optional_text(payload.tool_call_parser),
                            reasoning_parser=_normalize_optional_text(payload.reasoning_parser),
                            enable_thinking=payload.enable_thinking,
                            allow_long_context_override=payload.allow_long_context_override,
                            quantization=payload.quantization,
                            kv_cache_dtype=payload.kv_cache_dtype,
                        )
                except typer.Exit as exc:
                    runner_state["exit_code"] = getattr(exc, "exit_code", 1)
                except Exception as exc:  # pragma: no cover
                    runner_state["error"] = exc
                finally:
                    run_command_module.console = previous_console
                    stdout_stream.flush()
                    stderr_stream.flush()

            command_thread = Thread(target=_run_command, daemon=True)
            command_thread.start()

            known_container_status: str | None = None

            while command_thread.is_alive():
                try:
                    current_rows = list_managed_containers()
                    matching_rows = [row for row in current_rows if row.model_id == payload.model_id]
                    created_rows = [row for row in matching_rows if row.name not in before_model_names]
                    selected_row = None
                    if created_rows:
                        selected_row = sorted(created_rows, key=lambda row: row.name)[-1]
                    elif matching_rows:
                        selected_row = sorted(matching_rows, key=lambda row: row.name)[-1]

                    if selected_row is not None:
                        if selected_row.name != known_container_name:
                            known_container_name = selected_row.name
                            _set_task_container(task_id, selected_row.name)
                            _append_task_event(
                                task_id,
                                f"Container detected: {selected_row.name}.",
                            )

                        if selected_row.status != known_container_status:
                            known_container_status = selected_row.status
                            _append_task_event(
                                task_id,
                                f"Container status changed to `{selected_row.status}`.",
                            )
                except VaquilaError:
                    pass

                sleep(1.0)

            command_thread.join()

            try:
                after_rows = list_managed_containers()
                running_for_model = [
                    row
                    for row in after_rows
                    if row.model_id == payload.model_id and row.status == "running"
                ]
                created_candidates = [
                    row.name for row in running_for_model if row.name not in before_model_names
                ]
                if created_candidates:
                    selected_container_name = sorted(created_candidates)[-1]
                elif running_for_model:
                    selected_container_name = sorted([row.name for row in running_for_model])[-1]
            except VaquilaError:
                selected_container_name = None

            _set_task_container(task_id, selected_container_name)
            if selected_container_name is not None:
                _append_task_event(task_id, f"Container ready: {selected_container_name}.")

            runner_error = runner_state["error"]
            runner_exit_code = runner_state["exit_code"]
            if runner_error is not None:
                raise runner_error
            if runner_exit_code not in (None, 0):
                raise typer.Exit(code=runner_exit_code)

            _set_task(task_id, "succeeded", "Model is running.", finished=True)
        except typer.Exit as exc:
            code = getattr(exc, "exit_code", 1)
            failure_container_name = selected_container_name or known_container_name
            _append_container_logs_to_task(task_id, failure_container_name)
            _set_task(task_id, "failed", f"Run command failed (exit code={code}).", finished=True)
        except Exception as exc:  # pragma: no cover
            failure_container_name = selected_container_name or known_container_name
            _append_container_logs_to_task(task_id, failure_container_name)
            _set_task(task_id, "failed", str(exc), finished=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the Web UI page."""
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return "<h1>vAquila Web UI</h1><p>Missing frontend assets.</p>"

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Return application health status."""
        return {"status": "ok"}

    @app.get("/api/containers")
    def containers() -> dict[str, list[dict[str, object]]]:
        """Return active and stopped vAquila containers."""
        try:
            snapshot_by_gpu = read_all_gpu_snapshots()
        except VaquilaError:
            snapshot_by_gpu = None

        try:
            rows = list_managed_containers(snapshot_by_gpu=snapshot_by_gpu)
            return {"items": [asdict(row) for row in rows]}
        except VaquilaError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/gpu")
    def gpu_status() -> dict[str, object]:
        """Return GPU memory overview and per-model usage slices."""
        try:
            snapshots = read_all_gpu_snapshots()
        except VaquilaError:
            return {"available": False, "items": []}

        try:
            rows = list_managed_containers(snapshot_by_gpu=snapshots)
        except VaquilaError:
            rows = []

        items: list[dict[str, object]] = []
        for gpu_index in sorted(snapshots):
            snapshot = snapshots[gpu_index]
            model_rows: list[dict[str, object]] = []
            for row in rows:
                if row.gpu_index != gpu_index or row.status != "running":
                    continue
                model_rows.append(
                    {
                        "name": row.name,
                        "model_id": row.model_id,
                        "used_bytes": row.gpu_used_bytes,
                    }
                )

            items.append(
                {
                    "gpu_index": gpu_index,
                    "gpu_name": snapshot.name,
                    "total_bytes": snapshot.total_bytes,
                    "used_bytes": snapshot.used_bytes,
                    "free_bytes": snapshot.free_bytes,
                    "models": model_rows,
                }
            )

        return {"available": True, "items": items}

    @app.get("/api/logs/{container_name}")
    def container_logs(container_name: str, tail: int = 220) -> dict[str, object]:
        """Return recent logs for a vAquila container."""
        bounded_tail = max(20, min(tail, 1500))
        try:
            container = get_container(container_name)
            logs_text = container.logs(tail=bounded_tail).decode("utf-8", errors="replace")
            return {
                "container_name": container_name,
                "tail": bounded_tail,
                "logs": logs_text,
            }
        except VaquilaError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/cache")
    def cache_models() -> dict[str, list[dict[str, object]]]:
        """Return cached Hugging Face models with local sizes."""
        items = []
        for model_dir in list_cached_model_dirs():
            size_bytes = dir_size_bytes(model_dir)
            items.append(
                {
                    "model_id": cache_dir_to_model_id(model_dir),
                    "size_bytes": size_bytes,
                    "size_gib": size_bytes / (1024**3),
                    "path": str(model_dir),
                }
            )
        return {"items": items}

    @app.delete("/api/cache/{model_id:path}")
    def delete_cache(model_id: str) -> dict[str, object]:
        """Delete one model cache entry."""
        removed = _remove_model_cache_or_raise(model_id)
        return {"removed": removed}

    @app.get("/api/doctor")
    def doctor(gpu_index: int = 0) -> dict[str, list[dict[str, object]]]:
        """Return doctor-style checks as JSON."""
        checks: list[dict[str, object]] = []

        try:
            from vaquila.docker_service import check_docker_connection

            check_docker_connection()
            checks.append({"check": "Docker daemon", "ok": True, "details": "Connection OK"})
        except VaquilaError as exc:
            checks.append({"check": "Docker daemon", "ok": False, "details": str(exc)})

        try:
            snapshot = read_gpu_snapshot(gpu_index)
            checks.append(
                {
                    "check": "NVIDIA / NVML",
                    "ok": True,
                    "details": (
                        f"GPU {gpu_index} ({snapshot.name or 'Unknown'}) detected | "
                        f"total={snapshot.total_bytes / (1024**3):.2f} GiB "
                        f"free={snapshot.free_bytes / (1024**3):.2f} GiB"
                    ),
                }
            )
        except VaquilaError as exc:
            checks.append({"check": "NVIDIA / NVML", "ok": False, "details": str(exc)})

        try:
            cache_path = check_hf_cache_path()
            checks.append({"check": "Hugging Face cache", "ok": True, "details": cache_path})
        except VaquilaError as exc:
            checks.append({"check": "Hugging Face cache", "ok": False, "details": str(exc)})

        return {"checks": checks}

    @app.post("/api/run")
    def run_model(payload: RunRequest) -> dict[str, object]:
        """Queue and execute a model launch task."""
        task_id = str(uuid4())
        task = RunTask(
            id=task_id,
            model_id=payload.model_id,
            status="queued",
            message="Task queued.",
            container_name=None,
            started_at=_utc_now(),
            finished_at=None,
            events=[],
        )
        with tasks_lock:
            tasks[task_id] = task
            task_log_lines[task_id] = []

        _append_task_event(task_id, "Task queued.", status="queued")

        Thread(target=_launch_task, args=(task_id, payload), daemon=True).start()
        return {"task": asdict(task)}

    @app.get("/api/run/tasks")
    def list_run_tasks() -> dict[str, list[dict[str, object]]]:
        """List launch tasks in reverse chronological order."""
        with tasks_lock:
            items = sorted(tasks.values(), key=lambda item: item.started_at, reverse=True)
            return {"items": [asdict(item) for item in items]}

    @app.get("/api/run/tasks/{task_id}")
    def get_run_task(task_id: str) -> dict[str, object]:
        """Return a single run task."""
        with tasks_lock:
            task = tasks.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="Task not found.")
            return {"task": asdict(task)}

    @app.get("/api/run/tasks/{task_id}/logs")
    def get_run_task_logs(task_id: str) -> dict[str, object]:
        """Return accumulated logs for one launch task."""
        with tasks_lock:
            task = tasks.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="Task not found.")

            return {
                "task_id": task.id,
                "model_id": task.model_id,
                "container_name": task.container_name,
                "status": task.status,
                "events": task.events,
                "logs": "\n".join(task_log_lines.get(task_id, []))
                if task_log_lines.get(task_id)
                else "No task log available.",
            }

    @app.post("/api/stop")
    def stop_model(payload: StopRequest) -> dict[str, object]:
        """Stop one model and optionally purge cache."""
        removed_names: list[str] = []
        try:
            removed_names = stop_model_container(payload.model_id)
        except VaquilaError as exc:
            if not payload.purge_cache:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        cache_removed = False
        if payload.purge_cache:
            cache_removed = purge_model_cache(payload.model_id)

        return {"removed_containers": removed_names, "cache_removed": cache_removed}

    @app.post("/api/rm")
    def remove_cache(payload: RemoveRequest) -> dict[str, object]:
        """Remove one model from local cache if not running."""
        removed = _remove_model_cache_or_raise(payload.model_id)
        return {"removed": removed}

    @app.post("/api/infer")
    def infer(payload: InferRequest) -> dict[str, str]:
        """Run an inference call against a running model API."""
        try:
            answer = run_inference(
                base_url=payload.base_url,
                model_id=payload.model_id,
                prompt=payload.prompt,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                timeout_seconds=payload.timeout,
            )
            return {"answer": answer, "response": answer}
        except VaquilaError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
