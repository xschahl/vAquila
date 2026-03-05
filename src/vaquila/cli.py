"""Entrée CLI principale de vAquila."""

from __future__ import annotations

import typer

from vaquila.cli_commands import (
    cmd_doctor,
    cmd_infer,
    cmd_list_models,
    cmd_ps,
    cmd_rebalance,
    cmd_rm_model,
    cmd_run,
    cmd_stop,
)
from vaquila.config import CONFIG

app = typer.Typer(help="vAquila - Orchestration vLLM + Docker")


@app.command("list")
def list_models() -> None:
    """Liste les modèles présents dans le cache Hugging Face local."""
    cmd_list_models()


@app.command("rm")
def rm_model(
    model_id: str = typer.Argument(..., help="Model id Hugging Face à supprimer du cache"),
) -> None:
    """Supprime un modèle du cache Hugging Face local."""
    cmd_rm_model(model_id)


@app.command("run")
def run(
    model_id: str = typer.Argument(..., help="Model id Hugging Face, ex: meta-llama/Llama-3-8B-Instruct"),
    port: int = typer.Option(CONFIG.default_host_port, "--port", "-p", help="Port hôte exposé"),
    gpu_index: int = typer.Option(0, "--gpu", help="Index GPU NVIDIA"),
    buffer_gb: float = typer.Option(
        None,
        "--buffer-gb",
        help="Buffer VRAM en Gio réservé pour l'OS/processus tiers (optionnel, auto selon OS)",
    ),
    startup_timeout: int = typer.Option(
        900,
        "--startup-timeout",
        help="Timeout en secondes pour l'initialisation du modèle",
    ),
    rebalance_existing: bool = typer.Option(
        True,
        "--rebalance-existing/--no-rebalance-existing",
        help="Rééquilibre automatiquement les modèles déjà lancés sur ce GPU pour faire de la place",
    ),
    min_shared_ratio: float = typer.Option(
        0.25,
        "--min-shared-ratio",
        help="Ratio minimum par modèle en mode partagé (pré-check avant lancement vLLM)",
    ),
    max_num_seqs: int | None = typer.Option(
        None,
        "--max-num-seqs",
        help="Nombre de requêtes en parallèle (défaut: 1)",
    ),
    max_model_len: int | None = typer.Option(
        None,
        "--max-model-len",
        help="Contexte par utilisateur en tokens (défaut: 16384)",
    ),
    tool_call_parser: str | None = typer.Option(
        None,
        "--tool-call-parser",
        help="Tool call parser vLLM (défaut: aucun)",
    ),
    reasoning_parser: str | None = typer.Option(
        None,
        "--reasoning-parser",
        help="Reasoning parser vLLM (défaut: aucun)",
    ),
    enable_thinking: bool | None = typer.Option(
        None,
        "--enable-thinking/--disable-thinking",
        help="Active/désactive le mode thinking (si absent: question interactive, défaut=true)",
    ),
    allow_long_context_override: bool | None = typer.Option(
        None,
        "--allow-long-context-override/--no-allow-long-context-override",
        help="Autorise le dépassement de la limite de contexte modèle (risqué). Si absent: question au besoin.",
    ),
    share_gpu: bool = typer.Option(
        False,
        "--share-gpu",
        help="Autorise explicitement le mode partagé multi-modèles sur le même GPU",
    ),
) -> None:
    """Lance un modèle dans un conteneur vLLM en arrière-plan."""
    cmd_run(
        model_id=model_id,
        port=port,
        gpu_index=gpu_index,
        buffer_gb=buffer_gb,
        startup_timeout=startup_timeout,
        rebalance_existing=rebalance_existing,
        min_shared_ratio=min_shared_ratio,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        enable_thinking=enable_thinking,
        allow_long_context_override=allow_long_context_override,
        share_gpu=share_gpu,
    )


@app.command("ps")
def ps() -> None:
    """Liste les conteneurs vAquila actifs et leurs infos runtime."""
    cmd_ps()


@app.command("stop")
def stop(
    model_id: str = typer.Argument(..., help="Model id Hugging Face à arrêter"),
    purge_cache: bool = typer.Option(
        False,
        "--purge-cache",
        help="Supprime aussi le cache local Hugging Face du modèle après arrêt",
    ),
) -> None:
    """Stoppe et supprime le conteneur lié au modèle."""
    cmd_stop(model_id=model_id, purge_cache=purge_cache)


@app.command("rebalance")
def rebalance(
    gpu_index: int = typer.Option(0, "--gpu", help="Index GPU NVIDIA à rééquilibrer"),
    buffer_gb: float = typer.Option(
        None,
        "--buffer-gb",
        help="Buffer VRAM en Gio réservé pour l'OS/processus tiers (optionnel, auto selon OS)",
    ),
    startup_timeout: int = typer.Option(
        900,
        "--startup-timeout",
        help="Timeout en secondes pour l'initialisation des modèles",
    ),
    min_shared_ratio: float = typer.Option(
        0.25,
        "--min-shared-ratio",
        help="Ratio minimum par modèle en mode partagé (pré-check avant lancement vLLM)",
    ),
) -> None:
    """Rééquilibre les modèles déjà lancés sur un GPU en les relançant avec un ratio partagé."""
    cmd_rebalance(
        gpu_index=gpu_index,
        buffer_gb=buffer_gb,
        startup_timeout=startup_timeout,
        min_shared_ratio=min_shared_ratio,
    )


@app.command("doctor")
def doctor(
    gpu_index: int = typer.Option(0, "--gpu", help="Index GPU NVIDIA à vérifier"),
) -> None:
    """Vérifie l'environnement d'exécution (Docker, GPU, cache)."""
    cmd_doctor(gpu_index=gpu_index)


@app.command("infer")
def infer(
    model_id: str = typer.Argument(..., help="Model id ciblé par l'appel chat completion"),
    prompt: str = typer.Argument(..., help="Prompt utilisateur à envoyer au modèle"),
    base_url: str = typer.Option(
        CONFIG.inference_base_url,
        "--base-url",
        help="Base URL de l'API OpenAI-compatible vLLM",
    ),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Nombre maximal de tokens générés"),
    temperature: float = typer.Option(0.2, "--temperature", help="Température d'échantillonnage"),
    timeout: int = typer.Option(120, "--timeout", help="Timeout HTTP en secondes"),
) -> None:
    """Teste l'inférence d'un modèle déjà lancé via l'API vLLM."""
    cmd_infer(
        model_id=model_id,
        prompt=prompt,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
