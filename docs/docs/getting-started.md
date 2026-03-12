---
title: Getting Started
---

Choose the startup path that matches your machine and your target workflow.

## Choose a path

- [Classic setup with GPU support](./getting-started-gpu)
- [CPU-only setup without GPU](./getting-started-cpu-only)
  Both variants use the published image `ghcr.io/xschahl/vaquila:latest` and cover:

- simple `docker run`
- Dockerfile-based wrapper images
- Docker Compose
- CLI usage
- Web UI usage

## Shared prerequisites

- Docker Desktop or Docker Engine
- Python is not required for Docker-first workflows
- A host path for the Hugging Face cache that Docker can read

Set at least:

```bash
VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache
```

For CPU-only workflows, also set:

```bash
VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest
```

On Windows with Docker Desktop, prefer a path that the Docker daemon can read correctly.

## Quick check

```bash
docker pull ghcr.io/xschahl/vaquila:latest
docker run --rm ghcr.io/xschahl/vaquila:latest --help
```

## Examples

Functional examples are available in `docs/examples/`:

- `docs/examples/ghcr/docker-compose.yml`
- `docs/examples/ghcr/Dockerfile`
- `docs/examples/ghcr/.env.example`
- `docs/examples/webui/docker-compose.yml`
- `docs/examples/webui/Dockerfile`

Use these pages for the step-by-step launch guides:

- [Classic setup with GPU support](./getting-started-gpu)
- [CPU-only setup without GPU](./getting-started-cpu-only)
