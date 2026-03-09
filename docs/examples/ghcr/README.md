# GHCR Runtime Example

This example runs `vaq` directly from a published GHCR image.

## 1) Configure

Copy `.env.example` to `.env` and update `VAQ_HF_CACHE_HOST_PATH`.

## 2) Quick validation

```bash
docker compose run --rm vaq --help
```

## 3) Run a model

```bash
# GPU
docker compose run --rm vaq run Qwen/Qwen3-0.6B --gpu 0 --port 8000

# CPU
docker compose run --rm vaq run openai-community/gpt2 --device cpu --port 8000
```

## 4) Alternative wrapper Dockerfile

```dockerfile
ARG VAQ_IMAGE=ghcr.io/xschahl/vaquila:v0.1.0-beta.1
FROM ${VAQ_IMAGE}
```
