"""vAquila main CLI entrypoint."""

from __future__ import annotations

import typer

from vaquila.cli_commands import (
    cmd_doctor,
    cmd_infer,
    cmd_list_models,
    cmd_ps,
    cmd_rm_model,
    cmd_run,
    cmd_stop,
    cmd_ui,
)
from vaquila.config import CONFIG

app = typer.Typer(help="vAquila - Orchestration vLLM + Docker")


@app.command("list")
def list_models() -> None:
    """List models available in the local Hugging Face cache."""
    cmd_list_models()


@app.command("rm")
def rm_model(
    model_id: str = typer.Argument(..., help="Hugging Face model id to remove from cache"),
) -> None:
    """Remove a model from the local Hugging Face cache."""
    cmd_rm_model(model_id)


@app.command("run")
def run(
    model_id: str = typer.Argument(..., help="Hugging Face model id, e.g. meta-llama/Llama-3-8B-Instruct"),
    port: int = typer.Option(CONFIG.default_host_port, "--port", "-p", help="Exposed host port"),
    gpu_index: int = typer.Option(0, "--gpu", help="NVIDIA GPU index"),
    buffer_gb: float = typer.Option(
        None,
        "--buffer-gb",
        help="VRAM safety buffer in GiB reserved for OS/other processes (optional, auto by OS)",
    ),
    startup_timeout: int = typer.Option(
        900,
        "--startup-timeout",
        help="Model startup timeout in seconds",
    ),
    max_num_seqs: int | None = typer.Option(
        None,
        "--max-num-seqs",
        help="Maximum parallel requests (default: 1)",
    ),
    max_model_len: int | None = typer.Option(
        None,
        "--max-model-len",
        help="Per-request context length in tokens (default: 16384)",
    ),
    tool_call_parser: str | None = typer.Option(
        None,
        "--tool-call-parser",
        help="vLLM tool call parser (default: none)",
    ),
    reasoning_parser: str | None = typer.Option(
        None,
        "--reasoning-parser",
        help="vLLM reasoning parser (default: none)",
    ),
    enable_thinking: bool | None = typer.Option(
        None,
        "--enable-thinking/--disable-thinking",
        help="Enable/disable thinking mode (if omitted: interactive prompt, default=true)",
    ),
    allow_long_context_override: bool | None = typer.Option(
        None,
        "--allow-long-context-override/--no-allow-long-context-override",
        help="Allow context length above model limit (risky). If omitted: interactive prompt when needed.",
    ),
    quantization: str = typer.Option(
        "auto",
        "--quantization",
        help="vLLM quantization (auto, none, fp8, awq, gptq...).",
    ),
    kv_cache_dtype: str | None = typer.Option(
        None,
        "--kv-cache-dtype",
        help="KV cache dtype: fp16 or fp8 (if omitted: interactive prompt)",
    ),
) -> None:
    """Launch a model in a background vLLM container."""
    cmd_run(
        model_id=model_id,
        port=port,
        gpu_index=gpu_index,
        buffer_gb=buffer_gb,
        startup_timeout=startup_timeout,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        enable_thinking=enable_thinking,
        allow_long_context_override=allow_long_context_override,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
    )


@app.command("ps")
def ps() -> None:
    """List active vAquila containers and runtime details."""
    cmd_ps()


@app.command("stop")
def stop(
    model_id: str = typer.Argument(..., help="Hugging Face model id to stop"),
    purge_cache: bool = typer.Option(
        False,
        "--purge-cache",
        help="Also remove the model local Hugging Face cache after stop",
    ),
) -> None:
    """Stop and remove the container linked to a model."""
    cmd_stop(model_id=model_id, purge_cache=purge_cache)


@app.command("doctor")
def doctor(
    gpu_index: int = typer.Option(0, "--gpu", help="NVIDIA GPU index to validate"),
) -> None:
    """Validate runtime environment (Docker, GPU, cache)."""
    cmd_doctor(gpu_index=gpu_index)


@app.command("infer")
def infer(
    model_id: str = typer.Argument(..., help="Model id used for the chat completion call"),
    prompt: str = typer.Argument(..., help="User prompt sent to the model"),
    base_url: str = typer.Option(
        CONFIG.inference_base_url,
        "--base-url",
        help="Base URL of the vLLM OpenAI-compatible API",
    ),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Maximum number of generated tokens"),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature"),
    timeout: int = typer.Option(120, "--timeout", help="HTTP timeout in seconds"),
) -> None:
    """Test inference against an already running model through the vLLM API."""
    cmd_infer(
        model_id=model_id,
        prompt=prompt,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


@app.command("ui")
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface for the local Web UI server"),
    port: int = typer.Option(8787, "--port", help="Web UI HTTP port"),
) -> None:
    """Start the local Web UI for managing models and containers."""
    cmd_ui(host=host, port=port)
