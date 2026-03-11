"""Runtime resolution helpers and GPU ratio heuristics."""

from __future__ import annotations

import re
from math import ceil, floor

import typer

from vaquila.helpers.cache import resolve_model_config
from vaquila.exceptions import VaquilaError

_GIB = 1024**3
_VLLM_BLOCK_SIZE = 16


def normalize_optional_text(value: str | None) -> str | None:
    """Normalize an optional text field (empty -> None)."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def estimate_required_ratio(
    max_num_seqs: int,
    max_model_len: int,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool,
    kv_cache_dtype: str = "auto",
    quantization: str | None = None,
    model_id: str | None = None,
    total_vram_gb: float | None = None,
) -> float:
    """Estimate the minimum GPU ratio required by runtime settings."""
    context_factor = max_model_len / 16384
    heuristic_ratio = 0.02
    heuristic_ratio += 0.001 * max_num_seqs
    heuristic_ratio += 0.02 * context_factor
    if tool_call_parser:
        heuristic_ratio += 0.002
    if reasoning_parser:
        heuristic_ratio += 0.002
    if enable_thinking:
        heuristic_ratio += 0.001

    if kv_cache_dtype == "fp8":
        heuristic_ratio *= 0.78

    quantization_factor = {
        "fp8": 0.82,
        "awq": 0.72,
        "gptq": 0.72,
        "bitsandbytes": 0.75,
        "marlin": 0.75,
    }
    if quantization:
        heuristic_ratio *= quantization_factor.get(quantization, 0.90)

    ratio = heuristic_ratio

    if model_id and total_vram_gb and total_vram_gb > 0:
        analytic_ratio = estimate_required_ratio_from_model_profile(
            model_id=model_id,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            quantization=quantization,
            total_vram_gb=total_vram_gb,
        )
        if analytic_ratio is not None:
            # Prefer profile-based estimation for first-run precision.
            # Keep a small safety margin for allocator and block-size effects.
            conservative_analytic = (analytic_ratio * 1.08) + 0.004

            # Heuristics still matter when close to analytic value, but cap their impact.
            # This avoids large first-run over-allocation when heuristic is too pessimistic.
            if heuristic_ratio <= analytic_ratio * 1.15:
                ratio = max(conservative_analytic, heuristic_ratio)
            else:
                ratio = conservative_analytic

            # Never inflate beyond a bounded offset above analytic estimate.
            ratio = min(ratio, analytic_ratio + 0.06)

    return max(0.02, min(ratio, 0.90))


def estimate_required_ratio_from_model_profile(
    model_id: str,
    max_num_seqs: int,
    max_model_len: int,
    kv_cache_dtype: str,
    quantization: str | None,
    total_vram_gb: float,
    disk_size_bytes: int | None = None,
) -> float | None:
    """Estimate GPU ratio from model profile (weights + KV cache + runtime overhead)."""
    breakdown = estimate_vram_breakdown_from_model_profile(
        model_id=model_id,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization,
        disk_size_bytes=disk_size_bytes,
    )
    if breakdown is None:
        return None

    estimated_total_gb = breakdown["total_gb"]
    ratio = estimated_total_gb / total_vram_gb
    return max(0.02, min(ratio, 0.95))


def estimate_vram_breakdown_from_model_profile(
    model_id: str,
    max_num_seqs: int,
    max_model_len: int,
    kv_cache_dtype: str,
    quantization: str | None,
    disk_size_bytes: int | None = None,
) -> dict[str, object] | None:
    """Estimate VRAM components from model config using weights + KV cache + overhead."""
    payload = resolve_model_config(model_id)
    if not isinstance(payload, dict):
        return None

    profile = _extract_attention_profile(payload)
    if profile is None:
        return None

    hidden_size, num_layers, num_heads, num_kv_heads, head_dim = profile
    vocab_size = _read_positive_int(payload, "vocab_size")
    intermediate_size = _read_positive_int(payload, "intermediate_size", "d_inner", "ffn_dim")

    # --- KV cache with vLLM block-size alignment ---
    kv_bytes_per_elem = 1 if kv_cache_dtype == "fp8" else 2
    kv_token_bytes = 2 * num_layers * num_kv_heads * head_dim * kv_bytes_per_elem
    aligned_tokens_per_seq = ceil(max_model_len / _VLLM_BLOCK_SIZE) * _VLLM_BLOCK_SIZE
    kv_cache_bytes = kv_token_bytes * aligned_tokens_per_seq * max_num_seqs
    # Block table metadata overhead (~1%)
    block_table_bytes = int(kv_cache_bytes * 0.01)
    kv_cache_bytes += block_table_bytes
    kv_cache_gb = kv_cache_bytes / _GIB

    # --- Model weights ---
    estimated_params, estimation_source = _estimate_parameter_count(
        model_id=model_id,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        payload=payload,
        intermediate_size=intermediate_size,
        disk_size_bytes=disk_size_bytes,
        quantization=quantization,
    )
    weight_bytes_per_param = _bytes_per_param_for_quantization(quantization)
    weights_gb = (estimated_params * weight_bytes_per_param) / _GIB

    # --- Runtime overhead (model-size-aware) ---
    runtime_overhead_gb = _estimate_runtime_overhead_gb(
        weights_gb=weights_gb,
        kv_cache_gb=kv_cache_gb,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )
    total_gb = weights_gb + kv_cache_gb + runtime_overhead_gb

    # --- Estimation confidence ---
    confidence_map = {
        "config_explicit": "high",
        "model_name": "medium",
        "config_intermediate": "high",
        "config_architecture": "medium",
        "disk_size_fallback": "low",
    }

    return {
        "weights_gb": weights_gb,
        "kv_cache_gb": kv_cache_gb,
        "runtime_overhead_gb": runtime_overhead_gb,
        "total_gb": total_gb,
        "kv_token_bytes": float(kv_token_bytes),
        "estimation_source": estimation_source,
        "estimation_confidence": confidence_map.get(estimation_source, "medium"),
    }


def estimate_max_num_seqs_from_model_profile(
    model_id: str,
    max_model_len: int,
    kv_cache_dtype: str,
    quantization: str | None,
    available_vram_gb: float,
    disk_size_bytes: int | None = None,
) -> int | None:
    """Estimate the maximum sustainable max-num-seqs for a given VRAM budget."""
    if available_vram_gb <= 0:
        return None

    one_seq = estimate_vram_breakdown_from_model_profile(
        model_id=model_id,
        max_num_seqs=1,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization,
        disk_size_bytes=disk_size_bytes,
    )
    two_seq = estimate_vram_breakdown_from_model_profile(
        model_id=model_id,
        max_num_seqs=2,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization,
        disk_size_bytes=disk_size_bytes,
    )
    if one_seq is None or two_seq is None:
        return None

    fixed_gb = float(one_seq["weights_gb"]) + float(one_seq["runtime_overhead_gb"])
    per_seq_gb = max(0.000001, float(two_seq["kv_cache_gb"]) - float(one_seq["kv_cache_gb"]))
    remaining_gb = available_vram_gb - fixed_gb
    if remaining_gb <= 0:
        return 0

    return max(0, floor(remaining_gb / per_seq_gb))


def suggest_runtime_fallbacks_from_vram_budget(
    model_id: str,
    max_num_seqs: int,
    max_model_len: int,
    kv_cache_dtype: str,
    quantization: str | None,
    total_vram_gb: float,
    max_available_ratio: float,
    disk_size_bytes: int | None = None,
) -> dict[str, object]:
    """Build actionable fallback recommendations for a VRAM pre-check failure."""
    available_vram_gb = max(0.0, total_vram_gb * max_available_ratio)
    result: dict[str, object] = {
        "available_vram_gb": available_vram_gb,
        "current_breakdown": None,
        "estimated_max_num_seqs": None,
        "estimated_max_model_len": None,
        "quantization_suggestions": [],
    }

    breakdown = estimate_vram_breakdown_from_model_profile(
        model_id=model_id,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization,
        disk_size_bytes=disk_size_bytes,
    )
    if breakdown is None:
        return result

    result["current_breakdown"] = breakdown

    estimated_max_num_seqs = estimate_max_num_seqs_from_model_profile(
        model_id=model_id,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization,
        available_vram_gb=available_vram_gb,
        disk_size_bytes=disk_size_bytes,
    )
    result["estimated_max_num_seqs"] = estimated_max_num_seqs

    fixed_gb = breakdown["weights_gb"] + breakdown["runtime_overhead_gb"]
    if max_num_seqs > 0 and available_vram_gb > fixed_gb:
        per_token_gb = (breakdown["kv_cache_gb"] / max_num_seqs) / max(1, max_model_len)
        if per_token_gb > 0:
            max_model_len_fit = int((available_vram_gb - fixed_gb) / per_token_gb)
            result["estimated_max_model_len"] = max(0, max_model_len_fit)

    quant_candidates = ["fp8", "awq", "gptq", "bitsandbytes", None]
    quant_suggestions: list[tuple[str, float]] = []
    for candidate in quant_candidates:
        if candidate == quantization:
            continue

        ratio = estimate_required_ratio_from_model_profile(
            model_id=model_id,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            quantization=candidate,
            total_vram_gb=total_vram_gb,
        )
        if ratio is None:
            continue
        if ratio <= max_available_ratio:
            quant_suggestions.append((candidate or "none", ratio))

    quant_suggestions.sort(key=lambda item: item[1])
    result["quantization_suggestions"] = quant_suggestions[:3]
    return result


def _extract_attention_profile(payload: dict[str, object]) -> tuple[int, int, int, int, int] | None:
    """Extract core attention dimensions required for KV cache memory formulas."""
    hidden_size = _read_positive_int(payload, "hidden_size", "d_model", "n_embd")
    num_layers = _read_positive_int(payload, "num_hidden_layers", "n_layer")
    num_heads = _read_positive_int(payload, "num_attention_heads", "n_head")
    num_kv_heads = _read_positive_int(payload, "num_key_value_heads", "num_kv_heads")

    if hidden_size is None or num_layers is None or num_heads is None:
        return None

    if num_kv_heads is None or num_kv_heads <= 0:
        num_kv_heads = num_heads

    head_dim = _read_positive_int(payload, "head_dim")
    if head_dim is None:
        head_dim = max(1, hidden_size // num_heads)

    return hidden_size, num_layers, num_heads, num_kv_heads, head_dim


def _read_positive_int(payload: dict[str, object], *keys: str) -> int | None:
    """Read a strictly positive integer from multiple possible keys."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return None


def _estimate_parameter_count(
    model_id: str,
    hidden_size: int,
    num_layers: int,
    vocab_size: int | None,
    payload: dict[str, object],
    intermediate_size: int | None = None,
    disk_size_bytes: int | None = None,
    quantization: str | None = None,
) -> tuple[float, str]:
    """Estimate model parameter count, preferring explicit metadata when available.

    Return ``(estimated_params, estimation_source)``.
    """
    explicit_params = _read_positive_int(
        payload,
        "num_parameters",
        "n_parameters",
        "parameter_count",
        "params",
    )
    if explicit_params is not None:
        return float(explicit_params), "config_explicit"

    model_id_params = _extract_params_from_model_id(model_id)
    if model_id_params is not None:
        return model_id_params, "model_name"

    # Prefer intermediate_size-based formula for LLaMA/Qwen-style architectures
    if intermediate_size is not None and intermediate_size > 0:
        # Attention: Q, K, V, O projections = 4 * H^2
        # FFN (gate_proj, up_proj, down_proj) = 3 * H * I
        per_layer = 4.0 * (hidden_size ** 2) + 3.0 * hidden_size * intermediate_size
        transformer_params = float(num_layers) * per_layer
        embedding_params = 0.0
        if vocab_size and vocab_size > 0:
            embedding_params = 2.0 * vocab_size * hidden_size
        return transformer_params + embedding_params, "config_intermediate"

    # Classic transformer heuristic (12 * L * H^2)
    transformer_params = 12.0 * num_layers * (hidden_size ** 2)
    embedding_params = 0.0
    if vocab_size and vocab_size > 0:
        embedding_params = 2.0 * vocab_size * hidden_size
    arch_estimate = transformer_params + embedding_params

    # Disk-size fallback when architecture heuristic might be unreliable
    if disk_size_bytes is not None and disk_size_bytes > 0:
        bytes_per_param = _bytes_per_param_for_quantization(quantization)
        # ~5% of disk is non-weight files (tokenizer, config, etc.)
        weight_bytes = disk_size_bytes * 0.95
        disk_params = weight_bytes / bytes_per_param
        # Use disk estimate only if it differs significantly from architecture
        if arch_estimate > 0 and abs(disk_params - arch_estimate) / arch_estimate > 0.3:
            return disk_params, "disk_size_fallback"

    return arch_estimate, "config_architecture"


def _extract_params_from_model_id(model_id: str) -> float | None:
    """Extract parameter count from model names like ``Llama-3-8B`` or ``Qwen3-0.6B``."""
    match = re.search(r"(?:^|[-_/])([0-9]+(?:\.[0-9]+)?)\s*([bm])(?:$|[-_/])", model_id.lower())
    if match is None:
        return None

    value = float(match.group(1))
    suffix = match.group(2)
    multiplier = 1_000_000_000 if suffix == "b" else 1_000_000
    params = value * multiplier
    if params <= 0:
        return None
    return params


def _bytes_per_param_for_quantization(quantization: str | None) -> float:
    """Approximate bytes-per-parameter in memory by quantization mode."""
    if not quantization:
        return 2.0

    q = quantization.lower()
    if q == "fp8":
        return 1.0
    if q in {"awq", "gptq", "marlin"}:
        # 4-bit weights plus scales/metadata overhead in practice.
        return 0.56
    if q in {"bitsandbytes", "bnb", "int8"}:
        return 1.05
    return 1.4


def _estimate_runtime_overhead_gb(
    weights_gb: float,
    kv_cache_gb: float,
    num_layers: int = 32,
    hidden_size: int = 4096,
) -> float:
    """Estimate runtime overhead (CUDA context, PyTorch allocator, vLLM internals).

    Model-size-aware: scales with architecture dimensions instead of using
    a hard cap, so large models (70B+) get a proportionally larger budget.
    """
    # CUDA context baseline (driver + context init)
    cuda_context = 0.5

    # PyTorch memory allocator cache and fragmentation
    allocator = max(0.2, weights_gb * 0.04)

    # vLLM sampling / logits buffers (scales with hidden_size)
    vllm_buffers = max(0.15, (hidden_size / 4096) * 0.2)

    # Block table metadata (grows slowly with KV cache)
    block_metadata = kv_cache_gb * 0.01

    # Activation memory during forward pass
    activation = num_layers * (hidden_size / 4096) * 0.005

    total = cuda_context + allocator + vllm_buffers + block_metadata + activation
    return max(1.0, total)


def resolve_kv_cache_dtype(kv_cache_dtype: str | None) -> str:
    """Resolve a vLLM-compatible KV cache dtype from option or interactive prompt."""
    if kv_cache_dtype is None:
        choice = typer.prompt(
            "KV cache dtype [auto/bfloat16/fp8]",
            default="auto",
            show_default=True,
        ).strip().lower()
    else:
        choice = kv_cache_dtype.strip().lower()

    aliases = {
        "fp16": "auto",
        "float16": "auto",
        "bf16": "bfloat16",
    }
    resolved = aliases.get(choice, choice)

    if resolved not in {"auto", "bfloat16", "fp8"}:
        raise VaquilaError(
            "--kv-cache-dtype must be `auto`, `bfloat16`, or `fp8` "
            "(`fp16` is accepted as a legacy alias and mapped to `auto`)."
        )

    return resolved


def resolve_quantization_strategy(model_id: str, quantization: str | None) -> tuple[str | None, str]:
    """Resolve vLLM quantization: explicit value, none, or auto from model config."""
    requested = (quantization or "auto").strip().lower()
    if requested in {"", "auto"}:
        return _infer_quantization_from_model(model_id)
    if requested in {"none", "no", "false"}:
        return None, "none (disabled)"
    if requested == "fp4":
        return "awq", "awq (fp4 mapping)"
    return requested, requested


def _infer_quantization_from_model(model_id: str) -> tuple[str | None, str]:
    """Infer likely quantization (FP8/4-bit) from model config and model id."""
    model_id_l = model_id.lower()
    payload = resolve_model_config(model_id)

    if isinstance(payload, dict):
        quant_cfg = payload.get("quantization_config")
        if isinstance(quant_cfg, dict):
            quant_method = quant_cfg.get("quant_method") or quant_cfg.get("quantization_method")
            if isinstance(quant_method, str) and quant_method.strip():
                method = quant_method.strip().lower()
                if method == "fp4":
                    return "awq", "awq (auto from fp4 config)"
                return method, f"{method} (auto from config)"

            bits = quant_cfg.get("bits")
            if isinstance(bits, (int, float)):
                bit_value = int(bits)
                if bit_value <= 4:
                    return "awq", "awq (auto 4-bit)"
                if bit_value == 8:
                    return "fp8", "fp8 (auto 8-bit)"

            if quant_cfg.get("load_in_4bit") is True:
                return "awq", "awq (auto load_in_4bit)"

        torch_dtype = payload.get("torch_dtype")
        if isinstance(torch_dtype, str) and "float8" in torch_dtype.lower():
            return "fp8", "fp8 (auto torch_dtype)"

    fp8_tokens = ("fp8", "float8")
    fp4_tokens = ("fp4", "int4", "4bit", "awq", "gptq")
    if any(token in model_id_l for token in fp8_tokens):
        return "fp8", "fp8 (auto from model id)"
    if any(token in model_id_l for token in fp4_tokens):
        if "gptq" in model_id_l:
            return "gptq", "gptq (auto from model id)"
        return "awq", "awq (auto from model id)"

    return None, "none (auto: no quantization detected)"


def is_retryable_vram_error(message: str) -> bool:
    """Detect vLLM errors that justify increasing GPU ratio."""
    patterns = (
        "No available memory for the cache blocks",
        "Free memory on device",
        "less than desired GPU memory utilization",
        "available KV cache memory",
        "Try increasing `gpu_memory_utilization`",
        "estimated maximum model length",
    )
    return any(pattern in message for pattern in patterns)


def extract_kv_cache_memory_bounds(message: str) -> tuple[float, float] | None:
    """Extract (needed_gib, available_gib) from a vLLM KV cache error message."""
    pattern = re.compile(
        r"\(([0-9]+(?:\.[0-9]+)?)\s*GiB\s*KV cache is needed,\s*"
        r"which is larger than the available KV cache memory\s*"
        r"\(([0-9]+(?:\.[0-9]+)?)\s*GiB\)",
        re.IGNORECASE,
    )
    match = pattern.search(message)
    if not match:
        return None

    needed = float(match.group(1))
    available = float(match.group(2))
    if needed <= 0 or available <= 0:
        return None
    return needed, available


def suggest_ratio_from_kv_cache_error(
    failed_ratio: float,
    message: str,
    safety_margin: float = 1.06,
) -> float | None:
    """Suggest a minimum ratio from KV cache error metrics (needed/available)."""
    bounds = extract_kv_cache_memory_bounds(message)
    if bounds is None:
        return None

    needed_gib, available_gib = bounds
    suggested = failed_ratio * (needed_gib / available_gib) * safety_margin
    if suggested <= 0:
        return None
    return suggested


def ratio_candidates(min_ratio: float, max_ratio: float) -> list[float]:
    """Generate ratio steps between required minimum and available maximum."""
    if min_ratio > max_ratio:
        return []

    candidates = [round(min_ratio, 3)]
    steps = (0.03, 0.05, 0.08, 0.12, 0.18)
    for step in steps:
        candidate = round(min_ratio + step, 3)
        if candidate < max_ratio and candidate not in candidates:
            candidates.append(candidate)

    max_rounded = round(max_ratio, 3)
    if max_rounded not in candidates:
        candidates.append(max_rounded)

    return candidates


def resolve_run_runtime_settings(
    max_num_seqs: int | None,
    max_model_len: int | None,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    enable_thinking: bool | None,
) -> tuple[int, int, str | None, str | None, bool]:
    """Resolve runtime parameters from options or interactive prompts."""
    resolved_max_num_seqs = max_num_seqs
    if resolved_max_num_seqs is None:
        resolved_max_num_seqs = typer.prompt("Parallel requests (max-num-seqs)", default=1, type=int)

    resolved_max_model_len = max_model_len
    if resolved_max_model_len is None:
        resolved_max_model_len = typer.prompt("Per-request context length (max-model-len)", default=16384, type=int)

    resolved_tool_call_parser = normalize_optional_text(tool_call_parser)
    if tool_call_parser is None:
        prompted_tool_call_parser = typer.prompt(
            "Tool call parser (leave empty = none)",
            default="",
            show_default=False,
        )
        resolved_tool_call_parser = normalize_optional_text(prompted_tool_call_parser)

    resolved_reasoning_parser = normalize_optional_text(reasoning_parser)
    if reasoning_parser is None:
        prompted_reasoning_parser = typer.prompt(
            "Reasoning parser (leave empty = none)",
            default="",
            show_default=False,
        )
        resolved_reasoning_parser = normalize_optional_text(prompted_reasoning_parser)

    resolved_enable_thinking = enable_thinking
    if resolved_enable_thinking is None:
        resolved_enable_thinking = typer.confirm("Enable thinking mode?", default=True)

    if resolved_max_num_seqs <= 0:
        raise VaquilaError("max-num-seqs must be greater than 0.")
    if resolved_max_model_len <= 0:
        raise VaquilaError("max-model-len must be greater than 0.")

    return (
        resolved_max_num_seqs,
        resolved_max_model_len,
        resolved_tool_call_parser,
        resolved_reasoning_parser,
        resolved_enable_thinking,
    )
