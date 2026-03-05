# 🦅 vAquila

> **The Ollama developer experience, the vLLM production power.**

**vAquila** (accessible via the `vaq` command) is an open-source AI model inference manager. It combines the absolute simplicity of a CLI with the production performance of **vLLM** and the isolation of **Docker**, all with smart and automated GPU management.

## 🎯 The Problem

- **Ollama** is amazing for local testing, but its architecture shows its limits in production when handling multiple concurrent requests.
- **vLLM** is the undisputed king of production, but its deployment is a hassle (manual VRAM calculation, Docker volume management, _Out of Memory_ crashes).

## ✨ The Solution: vAquila

vAquila orchestrates everything for you. Like an eagle soaring over your infrastructure, it analyzes your GPU state in real-time, calculates the perfect memory ratio, and deploys the vLLM Docker container invisibly and securely.

### Planned Features (Roadmap)

- [ ] **Auto-VRAM**: Automatic calculation of the `--gpu-memory-utilization` flag via NVML to prevent crashes.
- [ ] **One-Click Deployment**: Download and run models via a simple `vaq run <hf-model>` command.
- [ ] **Docker Orchestration**: Invisible management of containers, exposed ports, and Hugging Face cache.
- [ ] **Web UI**: A local dashboard to monitor active models and live GPU usage.

## 🚀 Quickstart (Docker-first MVP)

### Prerequisites

- Docker daemon running
- NVIDIA drivers + GPU access enabled for Docker (`--gpus` support)

### 1) Configure environment

Copy `.env.example` to `.env`, then set at least:

```bash
VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/your/.cache/huggingface
```

> This path is used by vAquila when it launches vLLM containers.

### 2) Build vAquila image

```bash
docker compose build
```

### 3) Use CLI through Docker

Run a model:

```bash
docker compose run --rm vaq run meta-llama/Llama-3-8B-Instruct --port 8000 --gpu 0 --buffer-gb 1.5
```

At launch, `vaq run` asks (with defaults):

- parallel requests (`max-num-seqs`, default `1`)
- context per user (`max-model-len`, default `16384`)
- tool call parser (default none)
- reasoning parser (default none)
- enable thinking (default true)

You can also pass them directly as CLI options for non-interactive usage.
vAquila applies the minimum GPU ratio required by your runtime settings (instead of grabbing the maximum VRAM).
If requested settings exceed available VRAM, launch is refused before starting vLLM.
If KV cache is still insufficient at startup, vAquila auto-tries a few higher ratio steps until the smallest viable ratio is found.
vAquila also validates `max-model-len` against the model config when available (cache HF first, then Hub).
If requested context exceeds model limit, vAquila now proposes a choice:

- optimize with long-context override (risky, sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`),
- or automatically clamp to model limit.

By default, `vaq run` now waits for startup readiness and displays a live startup indicator (download/load phases from vLLM logs).
You can tune startup waiting with `--startup-timeout 1200`.
When launching a model on a GPU already in use, vAquila asks for confirmation before enabling shared GPU rebalance.
For non-interactive usage, pass `--share-gpu` to allow shared launch explicitly.
Use `--min-shared-ratio` (default `0.25`) to fail fast before vLLM startup when shared VRAM would be too low.

Manual rebalance of already running models on one GPU:

```bash
docker compose run --rm vaq rebalance --gpu 0 --min-shared-ratio 0.25
```

List managed containers:

```bash
docker compose run --rm vaq ps
```

Stop a model container:

```bash
docker compose run --rm vaq stop meta-llama/Llama-3-8B-Instruct
```

List downloaded models in local Hugging Face cache:

```bash
docker compose run --rm vaq list
```

Remove a downloaded model from local cache:

```bash
docker compose run --rm vaq rm meta-llama/Llama-3-8B-Instruct
```

To stop and also remove local Hugging Face cache for that model:

```bash
docker compose run --rm vaq stop meta-llama/Llama-3-8B-Instruct --purge-cache
```

Run environment preflight checks:

```bash
docker compose run --rm vaq doctor --gpu 0
```

Test model inference:

```bash
docker compose run --rm vaq infer Qwen/Qwen3-0.6B "Dis bonjour en français"
```

### Windows note

On Docker Desktop (Linux containers), prefer a daemon-readable Linux-style path in `.env` for `VAQ_HF_CACHE_HOST_PATH`.

### Project structure

```text
src/vaquila/
├─ cli.py             # Entry point Typer (routing des commandes)
├─ cli_commands.py    # Façade de compatibilité vers les modules de commandes
├─ cli_helpers.py     # Façade de compatibilité vers les modules helpers
├─ commands/
│  ├─ run.py          # Orchestration de lancement `vaq run`
│  ├─ system.py       # `ps`, `stop`, `rebalance`, `doctor`, `infer`
│  └─ cache.py        # `list`, `rm`
├─ helpers/
│  ├─ startup.py      # Parsing logs startup + attente readiness
│  ├─ runtime.py      # Heuristiques ratio + prompts runtime
│  ├─ cache.py        # Cache HF + résolution limites contexte
│  ├─ context.py      # Stratégie dépassement contexte
│  ├─ rebalance.py    # Plans/rebalance multi-modèles
│  └─ types.py        # Dataclasses partagées (LaunchPlan)
├─ config.py          # Runtime configuration
├─ docker_service.py  # Docker SDK orchestration
├─ gpu.py             # NVML GPU memory checks
├─ inference.py       # HTTP inference client for vLLM API
├─ models.py          # Internal dataclasses
└─ exceptions.py      # User-facing functional errors
```

### Current MVP behavior

- Auto VRAM ratio calculation from NVML with safety buffer
- Graceful CLI errors (Docker daemon / NVIDIA unavailable)
- Hugging Face cache mounted from `~/.cache/huggingface` to `/root/.cache/huggingface`
- vAquila-managed containers labeled for `ps` and `stop` workflows
- `ps` reads NVML snapshots across detected GPUs for consistent VRAM reporting

### Containerization files

- `Dockerfile`: multi-stage build for a clean distributable image
- `docker-compose.yml`: Docker-first execution of the `vaq` CLI
- `.env.example`: self-host oriented runtime configuration

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **CLI**: Typer
- **Orchestration**: Official Docker SDK for Python (`docker`)
- **Hardware**: `nvidia-ml-py` (module `pynvml`) for NVIDIA GPU monitoring
