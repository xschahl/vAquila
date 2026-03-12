---
title: Run a Model (CLI)
---

The base command is:

```bash
vaq run <model_id>
```

Docker-first example with the published image:

```bash
docker run --rm \
  --gpus all \
  -e VAQ_HF_CACHE_HOST_PATH=/absolute/path/to/huggingface/cache \
  -v /absolute/path/to/huggingface/cache:/root/.cache/huggingface \
  ghcr.io/xschahl/vaquila:latest \
  run Qwen/Qwen3-0.6B --gpu 0 --port 8000
```

## Minimal examples

GPU launch:

```bash
vaq run Qwen/Qwen3-0.6B --gpu 0 --port 8000
```

CPU-only launch:

```bash
vaq run openai-community/gpt2 --device cpu --port 8000
```

Manual mode without estimation or auto-tuning:

```bash
vaq run Qwen/Qwen3-0.6B --gpu 0 --port 8000 --gpu-utilization 0.72 --cpu-utilization 0.60
```

## Main arguments

- `model_id`: Hugging Face model id to launch, for example `Qwen/Qwen3-0.6B`.
- `--port` or `-p`: host port exposed by the vLLM API. The launch is blocked if the port is already in use.
- `--device gpu|cpu`: selects the compute backend. Use `gpu` for NVIDIA acceleration, `cpu` for CPU-only mode.
- `--gpu`: NVIDIA GPU index used in GPU mode.

## Runtime sizing arguments

- `--max-num-seqs`: maximum number of parallel requests handled by the runtime.
- `--max-model-len`: context length in tokens for each request.
- `--buffer-gb`: safety VRAM buffer reserved for the OS and other processes.
- `--startup-timeout`: startup timeout in seconds while vAquila waits for vLLM readiness.

These values directly affect memory usage. Larger values generally require more VRAM or RAM.

## Manual override arguments

- `--gpu-utilization`: manual GPU memory utilization ratio in `(0, 1]`.
- `--cpu-utilization`: manual CPU limit ratio in `(0, 1]`.

When one of these manual overrides is set, automatic estimation and optimization are bypassed for that launch.

## Model behavior arguments

- `--quantization`: runtime quantization strategy such as `auto`, `none`, `fp8`, `awq`, or `gptq`.
- `--kv-cache-dtype`: KV cache dtype such as `auto`, `bfloat16`, or `fp8`.
- `--tool-call-parser`: vLLM tool call parser.
- `--reasoning-parser`: vLLM reasoning parser.
- `--enable-thinking` or `--disable-thinking`: enables or disables thinking mode.
- `--allow-long-context-override` or `--no-allow-long-context-override`: allows a context length above the model limit when supported, which is a risky advanced override.

## How to choose values

- Start with the defaults if you do not already know the model requirements.
- Increase `--max-num-seqs` only when you need more parallel requests.
- Increase `--max-model-len` only when you need more context.
- Use manual utilization ratios only when you want to force a specific runtime budget.
- Prefer `--device cpu` only when GPU is unavailable or not desired.
