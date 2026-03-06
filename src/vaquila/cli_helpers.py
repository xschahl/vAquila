"""Compatibility facade for vAquila CLI helper utilities."""

from vaquila.helpers.cache import (
    cache_dir_to_model_id,
    check_hf_cache_path,
    dir_size_bytes,
    extract_model_context_limit,
    fetch_remote_model_config,
    format_gb,
    hub_cache_root,
    list_cached_model_dirs,
    model_cache_repo_dir,
    purge_model_cache,
    read_cached_model_config,
    resolve_model_config,
    resolve_model_context_limit,
)
from vaquila.helpers.context import resolve_context_strategy
from vaquila.helpers.rebalance import (
    compute_shared_ratio,
    estimate_shared_ratio_before_rebalance,
    launch_plan_from_container,
    rebalance_and_start,
)
from vaquila.helpers.runtime import (
    estimate_required_ratio,
    extract_kv_cache_memory_bounds,
    is_retryable_vram_error,
    normalize_optional_text,
    ratio_candidates,
    resolve_kv_cache_dtype,
    resolve_quantization_strategy,
    resolve_run_runtime_settings,
    suggest_ratio_from_kv_cache_error,
)
from vaquila.helpers.startup import (
    clean_log_line,
    extract_kv_max_concurrency,
    extract_root_error,
    extract_startup_hint,
    wait_until_model_ready,
)
from vaquila.helpers.types import LaunchPlan

__all__ = [
    "LaunchPlan",
    "cache_dir_to_model_id",
    "check_hf_cache_path",
    "clean_log_line",
    "compute_shared_ratio",
    "dir_size_bytes",
    "estimate_required_ratio",
    "estimate_shared_ratio_before_rebalance",
    "extract_kv_cache_memory_bounds",
    "extract_kv_max_concurrency",
    "extract_model_context_limit",
    "extract_root_error",
    "extract_startup_hint",
    "fetch_remote_model_config",
    "format_gb",
    "hub_cache_root",
    "is_retryable_vram_error",
    "launch_plan_from_container",
    "list_cached_model_dirs",
    "model_cache_repo_dir",
    "normalize_optional_text",
    "purge_model_cache",
    "ratio_candidates",
    "read_cached_model_config",
    "resolve_model_config",
    "resolve_kv_cache_dtype",
    "resolve_quantization_strategy",
    "rebalance_and_start",
    "resolve_context_strategy",
    "resolve_model_context_limit",
    "resolve_run_runtime_settings",
    "suggest_ratio_from_kv_cache_error",
    "wait_until_model_ready",
]
