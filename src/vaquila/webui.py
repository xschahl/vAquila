"""Local Web UI services for vAquila."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

import typer
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vaquila.cli_helpers import (
    cache_dir_to_model_id,
    check_hf_cache_path,
    dir_size_bytes,
    list_cached_model_dirs,
    purge_model_cache,
)
from vaquila.commands.run import cmd_run
from vaquila.config import CONFIG
from vaquila.docker_service import list_managed_containers, stop_model_container
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
    started_at: str
    finished_at: str | None


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
    kv_cache_dtype: str = "fp16"


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


def create_web_app() -> FastAPI:
    """Create and configure the FastAPI Web UI application."""
    app = FastAPI(title="vAquila Web UI", version="0.1.0")
    assets_dir = Path(__file__).resolve().parent / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    tasks: dict[str, RunTask] = {}
    tasks_lock = Lock()

    def _set_task(task_id: str, status: str, message: str, finished: bool = False) -> None:
        with tasks_lock:
            current = tasks.get(task_id)
            if current is None:
                return
            current.status = status
            current.message = message
            if finished:
                current.finished_at = _utc_now()

    def _launch_task(task_id: str, payload: RunRequest) -> None:
        _set_task(task_id, "running", "Launching model container...")
        try:
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
            _set_task(task_id, "succeeded", "Model is running.", finished=True)
        except typer.Exit as exc:
            code = getattr(exc, "exit_code", 1)
            _set_task(task_id, "failed", f"Run command failed (exit code={code}).", finished=True)
        except Exception as exc:  # pragma: no cover
            _set_task(task_id, "failed", str(exc), finished=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the Web UI page."""
        return _INDEX_HTML

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

    @app.get("/api/cache")
    def cache_models() -> dict[str, list[dict[str, object]]]:
        """Return cached Hugging Face models with local sizes."""
        models: list[dict[str, object]] = []
        for model_dir in list_cached_model_dirs():
            models.append(
                {
                    "model_id": cache_dir_to_model_id(model_dir),
                    "size_bytes": dir_size_bytes(model_dir),
                    "path": str(model_dir),
                }
            )
        return {"items": models}

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
                        f"GPU {gpu_index} detected | total={snapshot.total_bytes / (1024**3):.2f} GiB "
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
            started_at=_utc_now(),
            finished_at=None,
        )
        with tasks_lock:
            tasks[task_id] = task

        thread = Thread(target=_launch_task, args=(task_id, payload), daemon=True)
        thread.start()
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

        return {
            "removed_containers": removed_names,
            "cache_removed": cache_removed,
        }

    @app.post("/api/rm")
    def remove_cache(payload: RemoveRequest) -> dict[str, object]:
        """Remove one model from local cache if not running."""
        try:
            managed_containers = list_managed_containers()
        except VaquilaError:
            managed_containers = []

        running_for_model = [
            container
            for container in managed_containers
            if container.model_id == payload.model_id and container.status == "running"
        ]
        if running_for_model:
            names = ", ".join(container.name for container in running_for_model)
            raise HTTPException(
                status_code=400,
                detail=f"Model `{payload.model_id}` is still running ({names}). Stop it first.",
            )

        removed = purge_model_cache(payload.model_id)
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
            return {"answer": answer}
        except VaquilaError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


_INDEX_HTML = """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>vAquila Web UI</title>
    <link rel="icon" type="image/png" href="/assets/logo-base.png" />
    <style>
      :root {
        --bg: #f5f7fb;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --text: #0f172a;
        --text-soft: #475569;
        --border: #dbe3ee;
        --shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
        --primary: #2563eb;
        --primary-strong: #1d4ed8;
        --danger: #dc2626;
        --danger-strong: #b91c1c;
        --success-bg: #dcfce7;
        --success-fg: #166534;
        --error-bg: #fee2e2;
        --error-fg: #991b1b;
        --badge-default-bg: #e2e8f0;
        --badge-default-fg: #334155;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --bg: #0b1220;
          --surface: #121a2a;
          --surface-soft: #182133;
          --text: #e2e8f0;
          --text-soft: #94a3b8;
          --border: #223047;
          --shadow: 0 10px 28px rgba(2, 6, 23, 0.45);
          --primary: #3b82f6;
          --primary-strong: #2563eb;
          --danger: #ef4444;
          --danger-strong: #dc2626;
          --success-bg: #052e16;
          --success-fg: #86efac;
          --error-bg: #450a0a;
          --error-fg: #fca5a5;
          --badge-default-bg: #1e293b;
          --badge-default-fg: #cbd5e1;
        }
      }

      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Inter, Segoe UI, system-ui, -apple-system, sans-serif;
        background: radial-gradient(circle at top, #1d4ed810 0, transparent 40%), var(--bg);
        color: var(--text);
      }
      main {
        max-width: 1240px;
        margin: 0 auto;
        padding: 24px 16px 36px;
      }
      .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 16px;
      }
      .brand {
        display: flex;
        align-items: center;
        gap: 14px;
      }
      .brand-mark {
        width: 72px;
        height: 72px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, var(--surface), var(--surface-soft));
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18), 0 2px 8px rgba(15, 23, 42, 0.14);
        padding: 8px;
        object-fit: contain;
      }
      @media (max-width: 720px) {
        .brand-mark {
          width: 58px;
          height: 58px;
          border-radius: 14px;
          padding: 6px;
        }
      }
      .brand-dark { display: none; }
      .brand-light { display: block; }
      @media (prefers-color-scheme: dark) {
        .brand-dark { display: block; }
        .brand-light { display: none; }
      }
      h1 {
        margin: 0;
        font-size: 28px;
        letter-spacing: -0.02em;
      }
      .subhead {
        margin: 4px 0 0;
        color: var(--text-soft);
        font-size: 14px;
      }

      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
      }
      .stat {
        background: linear-gradient(180deg, var(--surface), var(--surface-soft));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        box-shadow: var(--shadow);
      }
      .stat .label { font-size: 12px; color: var(--text-soft); text-transform: uppercase; letter-spacing: .06em; }
      .stat .value { margin-top: 6px; font-size: 24px; font-weight: 700; }

      .layout-grid {
        display: grid;
        grid-template-columns: 1.15fr 1fr;
        gap: 14px;
      }
      @media (max-width: 1020px) {
        .layout-grid { grid-template-columns: 1fr; }
      }

      .stack { display: grid; gap: 14px; }
      section {
        background: linear-gradient(180deg, var(--surface), var(--surface-soft));
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        box-shadow: var(--shadow);
      }
      .section-title {
        margin: 0 0 12px;
        font-size: 16px;
        font-weight: 650;
      }

      .form-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
      }
      label {
        display: block;
        margin-bottom: 5px;
        font-size: 12px;
        color: var(--text-soft);
      }
      input, textarea, select {
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        padding: 9px 10px;
      }
      input:focus, textarea:focus, select:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px #2563eb22;
      }
      textarea { min-height: 110px; resize: vertical; }

      .row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
      }
      .row-between { justify-content: space-between; }

      button {
        border: 0;
        border-radius: 10px;
        padding: 9px 12px;
        font-weight: 600;
        color: #fff;
        background: var(--primary);
        cursor: pointer;
      }
      button:hover { background: var(--primary-strong); }
      button.secondary { background: #334155; }
      button.secondary:hover { background: #1e293b; }
      button.danger { background: var(--danger); }
      button.danger:hover { background: var(--danger-strong); }

      .table-wrap { overflow-x: auto; }
      table { width: 100%; border-collapse: collapse; min-width: 540px; }
      th, td {
        text-align: left;
        padding: 9px;
        border-bottom: 1px solid var(--border);
        font-size: 13px;
        vertical-align: top;
      }
      th {
        font-size: 12px;
        color: var(--text-soft);
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .status {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: var(--badge-default-bg);
        color: var(--badge-default-fg);
      }
      .status.running, .status.succeeded {
        background: #dcfce7;
        color: #166534;
      }
      .status.failed, .status.error, .status.exited {
        background: #fee2e2;
        color: #991b1b;
      }
      .status.queued, .status.created { background: #fef9c3; color: #854d0e; }

      .muted { color: var(--text-soft); font-size: 13px; }
      .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        word-break: break-all;
      }

      #toast {
        position: sticky;
        top: 10px;
        z-index: 10;
        margin-bottom: 12px;
        border-radius: 10px;
        padding: 10px 12px;
        display: none;
        border: 1px solid transparent;
      }
      #toast.ok {
        display: block;
        background: var(--success-bg);
        color: var(--success-fg);
        border-color: #22c55e55;
      }
      #toast.err {
        display: block;
        background: var(--error-bg);
        color: var(--error-fg);
        border-color: #ef444455;
      }

      #infer-output {
        margin: 0;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--surface);
        min-height: 70px;
        white-space: pre-wrap;
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <main>
      <header class=\"app-header\">
        <div class="brand">
          <img class="brand-mark brand-light" src="/assets/logo-light.png" alt="vAquila logo" />
          <img class="brand-mark brand-dark" src="/assets/logo-dark.png" alt="vAquila logo" />
          <div>
            <h1>vAquila Web UI</h1>
            <p class="subhead">Manage runtime, containers, cache, and inference from one local dashboard.</p>
          </div>
        </div>
        <button type=\"button\" class=\"secondary\" id=\"refresh-all\">Refresh all</button>
      </header>

      <div id=\"toast\"></div>

      <section class=\"summary-grid\">
        <div class=\"stat\"><div class=\"label\">Containers</div><div class=\"value\" id=\"stat-containers\">0</div></div>
        <div class=\"stat\"><div class=\"label\">Running</div><div class=\"value\" id=\"stat-running\">0</div></div>
        <div class=\"stat\"><div class=\"label\">Cached Models</div><div class=\"value\" id=\"stat-cache\">0</div></div>
        <div class=\"stat\"><div class=\"label\">Run Tasks</div><div class=\"value\" id=\"stat-tasks\">0</div></div>
      </section>

      <div class=\"layout-grid\">
        <div class=\"stack\">
          <section>
            <h2 class=\"section-title\">Run Model</h2>
            <form id=\"run-form\">
              <div class=\"form-grid\">
                <div><label>Model ID</label><input name=\"model_id\" required placeholder=\"Qwen/Qwen3-0.6B\" /></div>
                <div><label>Port</label><input name=\"port\" type=\"number\" value=\"8000\" min=\"1\" max=\"65535\" /></div>
                <div><label>GPU Index</label><input name=\"gpu_index\" type=\"number\" value=\"0\" min=\"0\" /></div>
                <div><label>Buffer (GiB, optional)</label><input name=\"buffer_gb\" type=\"number\" step=\"0.1\" /></div>
                <div><label>Startup Timeout (s)</label><input name=\"startup_timeout\" type=\"number\" value=\"900\" min=\"10\" /></div>
                <div><label>Max Num Seqs</label><input name=\"max_num_seqs\" type=\"number\" value=\"1\" min=\"1\" /></div>
                <div><label>Max Model Len</label><input name=\"max_model_len\" type=\"number\" value=\"16384\" min=\"1\" /></div>
                <div><label>Tool Call Parser</label><input name=\"tool_call_parser\" value=\"\" /></div>
                <div><label>Reasoning Parser</label><input name=\"reasoning_parser\" value=\"\" /></div>
                <div><label>Quantization</label><input name=\"quantization\" value=\"auto\" /></div>
                <div>
                  <label>KV Cache Dtype</label>
                  <select name=\"kv_cache_dtype\"><option value=\"fp16\">fp16</option><option value=\"fp8\">fp8</option></select>
                </div>
                <div>
                  <label>Enable Thinking</label>
                  <select name=\"enable_thinking\"><option value=\"true\">true</option><option value=\"false\">false</option></select>
                </div>
                <div>
                  <label>Allow Long Context Override</label>
                  <select name=\"allow_long_context_override\"><option value=\"false\">false</option><option value=\"true\">true</option></select>
                </div>
              </div>
              <div class=\"row\" style=\"margin-top: 10px;\">
                <button type=\"submit\">Launch</button>
              </div>
            </form>
          </section>

          <section>
            <div class=\"row row-between\">
              <h2 class=\"section-title\" style=\"margin-bottom: 0;\">Inference</h2>
              <span class=\"muted\">OpenAI-compatible endpoint</span>
            </div>
            <form id=\"infer-form\" class=\"form-grid\">
              <div><label>Model ID</label><input name=\"model_id\" required /></div>
              <div><label>Base URL</label><input name=\"base_url\" value=\"http://localhost:8000\" /></div>
              <div><label>Max Tokens</label><input name=\"max_tokens\" type=\"number\" value=\"128\" min=\"1\" /></div>
              <div><label>Temperature</label><input name=\"temperature\" type=\"number\" step=\"0.1\" value=\"0.2\" /></div>
              <div><label>Timeout</label><input name=\"timeout\" type=\"number\" value=\"120\" min=\"1\" /></div>
              <div style=\"grid-column: 1 / -1;\"><label>Prompt</label><textarea name=\"prompt\" required></textarea></div>
              <div style=\"grid-column: 1 / -1;\" class=\"row"><button type=\"submit\">Infer</button></div>
            </form>
            <p class=\"muted\" style=\"margin: 10px 0 6px;\">Response</p>
            <pre id=\"infer-output\"></pre>
          </section>
        </div>

        <div class=\"stack\">
          <section>
            <h2 class=\"section-title\">Run Tasks</h2>
            <div class=\"table-wrap\">
              <table>
                <thead><tr><th>Model</th><th>Status</th><th>Message</th><th>Started</th></tr></thead>
                <tbody id=\"tasks-body\"></tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 class=\"section-title\">Containers</h2>
            <div class=\"table-wrap\">
              <table>
                <thead><tr><th>Name</th><th>Model</th><th>Status</th><th>Port</th><th>GPU</th><th>Action</th></tr></thead>
                <tbody id=\"containers-body\"></tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 class=\"section-title\">Cached Models</h2>
            <div class=\"table-wrap\">
              <table>
                <thead><tr><th>Model</th><th>Size (GiB)</th><th>Path</th><th>Action</th></tr></thead>
                <tbody id=\"cache-body\"></tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </main>

    <script>
      const toast = document.getElementById('toast');
      const statContainers = document.getElementById('stat-containers');
      const statRunning = document.getElementById('stat-running');
      const statCache = document.getElementById('stat-cache');
      const statTasks = document.getElementById('stat-tasks');

      const showToast = (message, ok = true) => {
        toast.textContent = message;
        toast.className = ok ? 'ok' : 'err';
        setTimeout(() => {
          toast.style.display = 'none';
        }, 3200);
      };

      const statusClass = (value) => {
        const normalized = String(value || '').toLowerCase().trim();
        if (!normalized) return 'status';
        return `status ${normalized.replace(/[^a-z0-9_-]+/g, '-')}`;
      };

      const jsonFetch = async (url, options = {}) => {
        const response = await fetch(url, {
          headers: { 'Content-Type': 'application/json' },
          ...options,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.detail || 'Request failed');
        }
        return data;
      };

      const toGiB = (bytes) => (bytes / (1024 ** 3)).toFixed(2);

      const refreshContainers = async () => {
        const data = await jsonFetch('/api/containers');
        const body = document.getElementById('containers-body');
        body.innerHTML = '';

        let runningCount = 0;
        for (const item of data.items) {
          if (String(item.status).toLowerCase() === 'running') {
            runningCount += 1;
          }

          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class=\"mono\">${item.name}</td>
            <td>${item.model_id}</td>
            <td><span class=\"${statusClass(item.status)}\">${item.status}</span></td>
            <td>${item.host_port ?? 'n/a'}</td>
            <td>${item.gpu_index ?? 'n/a'}</td>
            <td><button class=\"danger\" data-model=\"${item.model_id}\">Stop</button></td>
          `;
          body.appendChild(tr);
        }

        statContainers.textContent = String(data.items.length);
        statRunning.textContent = String(runningCount);

        body.querySelectorAll('button[data-model]').forEach((btn) => {
          btn.addEventListener('click', async () => {
            try {
              await jsonFetch('/api/stop', {
                method: 'POST',
                body: JSON.stringify({ model_id: btn.dataset.model, purge_cache: false }),
              });
              showToast('Model stopped.');
              await refreshAll();
            } catch (err) {
              showToast(err.message, false);
            }
          });
        });
      };

      const refreshCache = async () => {
        const data = await jsonFetch('/api/cache');
        const body = document.getElementById('cache-body');
        body.innerHTML = '';

        for (const item of data.items) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${item.model_id}</td>
            <td>${toGiB(item.size_bytes)}</td>
            <td class=\"mono\">${item.path}</td>
            <td><button class=\"danger\" data-rm=\"${item.model_id}\">Remove</button></td>
          `;
          body.appendChild(tr);
        }

        statCache.textContent = String(data.items.length);

        body.querySelectorAll('button[data-rm]').forEach((btn) => {
          btn.addEventListener('click', async () => {
            try {
              const payload = { model_id: btn.dataset.rm };
              const result = await jsonFetch('/api/rm', { method: 'POST', body: JSON.stringify(payload) });
              if (!result.removed) {
                showToast('Model not found in cache.', false);
              } else {
                showToast('Model cache removed.');
              }
              await refreshAll();
            } catch (err) {
              showToast(err.message, false);
            }
          });
        });
      };

      const refreshTasks = async () => {
        const data = await jsonFetch('/api/run/tasks');
        const body = document.getElementById('tasks-body');
        body.innerHTML = '';

        const visibleTasks = data.items.slice(0, 8);
        for (const item of visibleTasks) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${item.model_id}</td>
            <td><span class=\"${statusClass(item.status)}\">${item.status}</span></td>
            <td>${item.message}</td>
            <td>${new Date(item.started_at).toLocaleString()}</td>
          `;
          body.appendChild(tr);
        }

        statTasks.textContent = String(data.items.length);
      };

      const refreshAll = async () => {
        await Promise.all([refreshContainers(), refreshCache(), refreshTasks()]);
      };

      document.getElementById('run-form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(event.target);
        const payload = {
          model_id: String(formData.get('model_id')).trim(),
          port: Number(formData.get('port')),
          gpu_index: Number(formData.get('gpu_index')),
          buffer_gb: formData.get('buffer_gb') ? Number(formData.get('buffer_gb')) : null,
          startup_timeout: Number(formData.get('startup_timeout')),
          max_num_seqs: Number(formData.get('max_num_seqs')),
          max_model_len: Number(formData.get('max_model_len')),
          tool_call_parser: String(formData.get('tool_call_parser') || ''),
          reasoning_parser: String(formData.get('reasoning_parser') || ''),
          enable_thinking: String(formData.get('enable_thinking')) === 'true',
          allow_long_context_override: String(formData.get('allow_long_context_override')) === 'true',
          quantization: String(formData.get('quantization') || 'auto'),
          kv_cache_dtype: String(formData.get('kv_cache_dtype') || 'fp16'),
        };

        try {
          await jsonFetch('/api/run', { method: 'POST', body: JSON.stringify(payload) });
          showToast('Run task created.');
          await refreshTasks();
        } catch (err) {
          showToast(err.message, false);
        }
      });

      document.getElementById('infer-form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(event.target);
        const payload = {
          model_id: String(formData.get('model_id')).trim(),
          prompt: String(formData.get('prompt')).trim(),
          base_url: String(formData.get('base_url')).trim(),
          max_tokens: Number(formData.get('max_tokens')),
          temperature: Number(formData.get('temperature')),
          timeout: Number(formData.get('timeout')),
        };

        try {
          const data = await jsonFetch('/api/infer', { method: 'POST', body: JSON.stringify(payload) });
          document.getElementById('infer-output').textContent = data.answer;
          showToast('Inference completed.');
        } catch (err) {
          showToast(err.message, false);
        }
      });

      document.getElementById('refresh-all').addEventListener('click', refreshAll);

      refreshAll().catch((err) => showToast(err.message, false));
      setInterval(() => refreshTasks().catch(() => {}), 3500);
      setInterval(() => refreshContainers().catch(() => {}), 6000);
    </script>
  </body>
</html>
"""
