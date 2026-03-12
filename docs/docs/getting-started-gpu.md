---
title: Getting Started with GPU
---

This is the standard vAquila setup when your host has NVIDIA GPU support available for Docker.

## Prerequisites

- Docker Desktop or Docker Engine
- NVIDIA drivers
- Docker GPU support (`--gpus all`)

## Configure

Set a daemon-readable Hugging Face cache path:

```bash
VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache
VAQ_VLLM_IMAGE=vllm/vllm-openai:latest
VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest
```

## Use the official latest image directly

```bash
docker pull ghcr.io/xschahl/vaquila:latest
docker run --rm ghcr.io/xschahl/vaquila:latest --help
```

### CLI with GPU

```bash
docker run --rm \
  --gpus all \
  -e VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache \
  -v /absolute/path/to/huggingface/cache:/root/.cache/huggingface \
  ghcr.io/xschahl/vaquila:latest \
  run Qwen/Qwen3-0.6B --gpu 0 --port 8000
```

### Web UI with GPU

Use this mode when you want GPU telemetry in the UI and the ability to launch GPU-backed models from the Web UI:

```bash
docker run --rm \
  --gpus all \
  -p 8787:8787 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache \
  -e VAQ_VLLM_IMAGE=vllm/vllm-openai:latest \
  -e VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /absolute/path/to/huggingface/cache:/root/.cache/huggingface \
  ghcr.io/xschahl/vaquila:latest \
  ui --host 0.0.0.0 --port 8787
```

Then open `http://localhost:8787` in your browser.

## Use the latest image as a Dockerfile base

```dockerfile
ARG VAQ_IMAGE=ghcr.io/xschahl/vaquila:latest
FROM ${VAQ_IMAGE}
```

Build it:

```bash
docker build -t vaquila-local .
```

Run the CLI:

```bash
docker run --rm vaquila-local --help
```

Run the Web UI:

```bash
docker run --rm -p 8787:8787 vaquila-local ui --host 0.0.0.0 --port 8787
```

## Use Docker Compose with the latest image

### Compose for CLI usage

```yaml
services:
  vaq:
    image: ghcr.io/xschahl/vaquila:latest
    volumes:
      - ${VAQ_HF_CACHE_HOST_PATH}:/root/.cache/huggingface
    environment:
      VAQ_HF_CACHE_HOST_PATH: ${VAQ_HF_CACHE_HOST_PATH}
      VAQ_VLLM_IMAGE: vllm/vllm-openai:latest
      VAQ_VLLM_CPU_IMAGE: vllm/vllm-openai-cpu:latest
```

Run a GPU launch:

```bash
docker compose run --rm vaq run Qwen/Qwen3-0.6B --gpu 0 --port 8000
```

### Compose for Web UI with GPU

```yaml
services:
  vaq-webui:
    image: ghcr.io/xschahl/vaquila:latest
    gpus: all
    command: ["ui", "--host", "0.0.0.0", "--port", "8787"]
    ports:
      - "8787:8787"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ${VAQ_HF_CACHE_HOST_PATH}:/root/.cache/huggingface
    environment:
      VAQ_HF_CACHE_HOST_PATH: ${VAQ_HF_CACHE_HOST_PATH}
      VAQ_VLLM_IMAGE: vllm/vllm-openai:latest
      VAQ_VLLM_CPU_IMAGE: vllm/vllm-openai-cpu:latest
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
```

Start the UI:

```bash
docker compose up -d
```

## Example directories

- `docs/examples/ghcr/docker-compose.yml`
- `docs/examples/ghcr/Dockerfile`
- `docs/examples/webui/docker-compose.yml`
