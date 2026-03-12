---
title: Getting Started CPU-only
---

Use this path when your host has no NVIDIA GPU support or when you want to run vAquila only with CPU-backed models.

## Prerequisites

- Docker Desktop or Docker Engine
- No GPU runtime is required

## Configure

Set a daemon-readable Hugging Face cache path and a CPU-compatible vLLM image:

```bash
VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache
VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest
```

## Use the official latest image directly

```bash
docker pull ghcr.io/xschahl/vaquila:latest
docker run --rm ghcr.io/xschahl/vaquila:latest --help
```

### CLI with CPU only

```bash
docker run --rm \
  -e VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache \
  -e VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest \
  -v /absolute/path/to/huggingface/cache:/root/.cache/huggingface \
  ghcr.io/xschahl/vaquila:latest \
  run openai-community/gpt2 --device cpu --port 8000
```

### Web UI without GPU

GPU monitoring will be unavailable in the UI, and GPU launches are not available.

```bash
docker run --rm \
  -p 8787:8787 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache \
  -e VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest \
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
      VAQ_VLLM_CPU_IMAGE: vllm/vllm-openai-cpu:latest
```

Run a CPU-only launch:

```bash
docker compose run --rm vaq run openai-community/gpt2 --device cpu --port 8000
```

### Compose for Web UI without GPU

```yaml
services:
  vaq-webui:
    image: ghcr.io/xschahl/vaquila:latest
    command: ["ui", "--host", "0.0.0.0", "--port", "8787"]
    ports:
      - "8787:8787"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ${VAQ_HF_CACHE_HOST_PATH}:/root/.cache/huggingface
    environment:
      VAQ_HF_CACHE_HOST_PATH: ${VAQ_HF_CACHE_HOST_PATH}
      VAQ_VLLM_CPU_IMAGE: vllm/vllm-openai-cpu:latest
```

Start the UI:

```bash
docker compose up -d
```

## Example directories

- `docs/examples/ghcr/docker-compose.yml`
- `docs/examples/webui/docker-compose.yml`
