---
title: Getting Started
---

## Prerequisites

- Docker Desktop or Docker Engine
- NVIDIA stack for GPU mode
- Python is not required when using Docker-first workflows

## Configure

Copy `.env.example` to `.env` and set:

```bash
VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache
```

Optional images:

```bash
VAQ_VLLM_IMAGE=vllm/vllm-openai:latest
VAQ_VLLM_CPU_IMAGE=vllm/vllm-openai-cpu:latest-x86_64
```

## Build

```bash
docker compose build vaq
```

## Run a model

GPU mode:

```bash
docker compose run --rm vaq run Qwen/Qwen3-0.6B --gpu 0 --port 8000
```

CPU mode:

```bash
docker compose run --rm vaq run openai-community/gpt2 --device cpu --port 8000
```
