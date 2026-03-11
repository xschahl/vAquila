# vAquila Web UI Example

This directory provides examples of how to run the **vAquila Web UI** using Docker.

## Overview

vAquila includes a Web UI (powered by FastAPI and Uvicorn) that allows you to manage models and containers easily. To make this accessible from outside the Docker container, we must run it bound to `0.0.0.0` on a specific port (default is `8787`).

There are two primary ways to run the Web UI via Docker:

### Method 1: Using the Official Beta Image (Recommended)

The official image (`ghcr.io/xschahl/vaquila:v0.1.0-beta.1`) already contains all necessary dependencies and is based on a lightweight Debian environment (`python:3.11-slim`).

In the provided `docker-compose.yml`, this is the default configuration. We simply override the container command to start the UI:

```yaml
command: ["ui", "--host", "0.0.0.0", "--port", "8787"]
```

To run this:

1. Copy `.env.example` to `.env` (at the root of the project or alongside this compose file) and configure your `VAQ_HF_CACHE_HOST_PATH`.
   Also set `VAQ_VLLM_CPU_IMAGE` to a CPU-compatible vLLM image (for example `vllm/vllm-openai-cpu:latest`)
   so `--device cpu` never falls back to a GPU-only runtime image.
2. Run Compose:
   `docker compose up -d`
3. Access the UI in your browser at [http://localhost:8787](http://localhost:8787).

### Method 2: Building a Debian Environment From Scratch

If you have specific requirements, need additional OS dependencies, or want to compile the environment yourself, you can use the provided `Dockerfile`. This Dockerfile uses `python:3.11-slim` as the base Debian image and `pip install`s the package.

To use this method:

1. Open `docker-compose.yml`.
2. Comment out the `vaq-webui` service and uncomment the `vaq-webui-custom` service section.
3. Run Compose, forcing a build:
   `docker compose up -d --build`
4. Access the UI in your browser at [http://localhost:8787](http://localhost:8787).

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit** (for GPU access, `gpus: all`)
- Proper `.env` configuration (specifically `VAQ_HF_CACHE_HOST_PATH` so vLLM containers can share cached model weights).
