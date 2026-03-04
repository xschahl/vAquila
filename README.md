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

List managed containers:

```bash
docker compose run --rm vaq ps
```

Stop a model container:

```bash
docker compose run --rm vaq stop meta-llama/Llama-3-8B-Instruct
```

Run environment preflight checks:

```bash
docker compose run --rm vaq doctor --gpu 0
```

### Windows note

On Docker Desktop (Linux containers), prefer a daemon-readable Linux-style path in `.env` for `VAQ_HF_CACHE_HOST_PATH`.

### Project structure

```text
src/vaquila/
├─ cli.py             # Typer commands: run, ps, stop, doctor
├─ config.py          # Runtime configuration
├─ docker_service.py  # Docker SDK orchestration
├─ gpu.py             # NVML GPU memory checks
├─ models.py          # Internal dataclasses
└─ exceptions.py      # User-facing functional errors
```

### Current MVP behavior

- Auto VRAM ratio calculation from NVML with safety buffer
- Graceful CLI errors (Docker daemon / NVIDIA unavailable)
- Hugging Face cache mounted from `~/.cache/huggingface` to `/root/.cache/huggingface`
- vAquila-managed containers labeled for `ps` and `stop` workflows

### Containerization files

- `Dockerfile`: multi-stage build for a clean distributable image
- `docker-compose.yml`: Docker-first execution of the `vaq` CLI
- `.env.example`: self-host oriented runtime configuration

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **CLI**: Typer
- **Orchestration**: Official Docker SDK for Python (`docker`)
- **Hardware**: `nvidia-ml-py` (module `pynvml`) for NVIDIA GPU monitoring
