# 🦅 vAquila

<p align="center">
	<img src="src/vaquila/assets/logo-base.png" alt="vAquila logo" width="160" />
</p>

> **The Ollama developer experience, the vLLM production power.**

**vAquila** (accessible via the `vaq` command) is an open-source AI model inference manager. It combines the absolute simplicity of a CLI with the production performance of **vLLM** and the isolation of **Docker**, all with smart and automated GPU management.

## 🎯 The Problem

- **Ollama** is amazing for local testing, but its architecture shows its limits in production when handling multiple concurrent requests.
- **vLLM** is the undisputed king of production, but its deployment is a hassle (manual VRAM calculation, Docker volume management, _Out of Memory_ crashes).

## ✨ The Solution: vAquila

vAquila orchestrates everything for you. Like an eagle soaring over your infrastructure, it analyzes your GPU state in real-time, calculates the perfect memory ratio, and deploys the vLLM Docker container invisibly and securely.

### Planned Features (Roadmap)

- [x] **Auto-VRAM**: Automatic calculation of the `--gpu-memory-utilization` flag via NVML to prevent crashes.
- [x] **One-Click Deployment**: Download and run models via a simple `vaq run <hf-model>` command.
- [x] **Docker Orchestration**: Invisible management of containers, exposed ports, and Hugging Face cache.
- [x] **Web UI**: A local dashboard to manage models, containers, cache, and inference.

## 🚀 Quickstart (Docker-first MVP)

### Prerequisites

- Docker daemon running
- NVIDIA drivers + GPU access enabled for Docker (`--gpus` support) when using `--device gpu` (default)

### 1) Configure environment

Copy `.env.example` to `.env`, then set at least:

```bash
VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/your/.cache/huggingface
```

Optional runtime images:

```bash
# GPU runtime image (default)
VAQ_VLLM_IMAGE=vllm/vllm-openai:latest

# CPU runtime image used when --device cpu is selected
VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest-x86_64
```

For non-x86_64 hosts (ARM, Apple Silicon, etc.), use an architecture-specific CPU tag
or a custom-built CPU image from vLLM's CPU Docker instructions.

> This path is used by vAquila when it launches vLLM containers.

### 2) Build vAquila image

```bash
docker compose build
```

### 3) Use CLI through Docker

Run a model:

```bash
docker compose run --rm vaq run meta-llama/Llama-3-8B-Instruct --port 8000 --gpu 0 --buffer-gb 1.5

# CPU mode (no GPU allocation)
docker compose run --rm vaq run openai-community/gpt2 --device cpu --port 8000

# Manual utilization mode (skip estimation/optimization)
docker compose run --rm vaq run Qwen/Qwen3-0.6B --gpu 0 --gpu-utilization 0.72 --cpu-utilization 0.60
```

At launch, `vaq run` asks (with defaults):

- parallel requests (`max-num-seqs`, default `1`)
- context per user (`max-model-len`, default `16384`)
- tool call parser (default none)
- reasoning parser (default none)
- enable thinking (default true)
- quantization strategy (default `auto`)
- KV cache dtype (`auto`, `bfloat16`, or `fp8`; legacy `fp16` is mapped to `auto`)

You can also pass them directly as CLI options for non-interactive usage.
`vaq run` now supports `--device gpu|cpu` (default: `gpu`).
`vaq run` also supports manual overrides:

- `--gpu-utilization <ratio>` where ratio is in `(0, 1]`
- `--cpu-utilization <ratio>` where ratio is in `(0, 1]`

When one of these manual utilization options is provided, vAquila bypasses auto-estimation and optimization.
Both modes keep the same OpenAI-compatible API surface (`/v1/...`) because they still use a vLLM OpenAI server image.
vAquila computes a VRAM-aware initial GPU ratio from your runtime settings and model profile (weights + KV cache estimate + overhead).
If requested settings exceed available VRAM, launch is refused before starting vLLM.
If KV cache is still insufficient at startup, vAquila adjusts ratio with data-driven retries using vLLM error metrics (`needed GiB` / `available KV cache memory`) to converge faster.
vAquila also validates `max-model-len` against model config when available (HF cache first, then Hub).
If requested context exceeds model limit, vAquila proposes:

- optimize with long-context override (risky, sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`),
- or automatically clamp to model limit.

By default, `vaq run` waits for startup readiness and shows a live startup indicator (download/load phases parsed from vLLM logs).
You can tune startup waiting with `--startup-timeout 1200`.

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
docker compose run --rm vaq infer Qwen/Qwen3-0.6B "Say hello in English"
```

Start the local Web UI:

```bash
docker compose run --rm -p 8787:8787 vaq ui --host 0.0.0.0 --port 8787
```

Then open `http://localhost:8787` in your browser.

### Windows note

On Docker Desktop (Linux containers), prefer a daemon-readable Linux-style path in `.env` for `VAQ_HF_CACHE_HOST_PATH`.

### Project structure

```text
src/vaquila/
├─ cli.py             # Typer CLI entrypoint (command routing)
├─ cli_commands.py    # Compatibility facade for command modules
├─ cli_helpers.py     # Compatibility facade for helper modules
├─ commands/
│  ├─ run.py          # `vaq run` launch orchestration and GPU ratio tuning
│  ├─ system.py       # `ps`, `stop`, `doctor`, `infer`
│  └─ cache.py        # `list`, `rm`
├─ helpers/
│  ├─ startup.py      # Startup log parsing + readiness waiting
│  ├─ runtime.py      # Ratio heuristics + runtime prompts
│  ├─ cache.py        # HF cache helpers + context limit resolution
│  ├─ context.py      # Long-context strategy
│  ├─ rebalance.py    # Legacy multi-model rebalance helpers
│  └─ types.py        # Shared dataclasses (LaunchPlan)
├─ config.py          # Runtime configuration
├─ docker_service.py  # Docker SDK orchestration
├─ gpu.py             # NVML GPU memory checks
├─ inference.py       # HTTP inference client for vLLM API
├─ models.py          # Internal dataclasses
└─ exceptions.py      # User-facing functional errors
```

### Current MVP behavior

- Auto VRAM ratio calculation from NVML with safety buffer
- Profile-informed initial ratio (model config + quantization + KV cache estimate)
- KV-aware tuning from vLLM runtime errors (`needed` vs `available` KV cache memory)
- Graceful CLI errors (Docker daemon / NVIDIA unavailable)
- Hugging Face cache mounted from `~/.cache/huggingface` to `/root/.cache/huggingface`
- vAquila-managed containers labeled for `ps` and `stop` workflows
- `ps` reads NVML snapshots across detected GPUs for consistent VRAM reporting
- Local Web UI for launch, monitoring, stop/remove, and inference workflows

### Containerization files

- `Dockerfile`: multi-stage build for a clean distributable image
- `docker-compose.yml`: Docker-first execution of the `vaq` CLI
- `.env.example`: self-host oriented runtime configuration

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **CLI**: Typer
- **Orchestration**: Official Docker SDK for Python (`docker`)
- **Hardware**: `nvidia-ml-py` (module `pynvml`) for NVIDIA GPU monitoring
