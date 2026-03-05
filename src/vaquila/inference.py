"""Services d'appel HTTP à l'API vLLM OpenAI-compatible."""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vaquila.exceptions import VaquilaError


def _sanitize_model_output(text: str) -> str:
    """Supprime les balises de raisonnement éventuelles pour un rendu utilisateur propre."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text.strip()


def run_inference(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
    """Exécute une requête de chat completion et renvoie la réponse assistant."""
    if max_tokens <= 0:
        raise VaquilaError("max_tokens doit être supérieur à 0.")
    if timeout_seconds <= 0:
        raise VaquilaError("timeout doit être supérieur à 0.")

    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode("utf-8")

    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_data = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise VaquilaError(f"Échec API vLLM ({exc.code}): {details}") from exc
    except URLError as exc:
        raise VaquilaError(
            f"Impossible de joindre l'API vLLM sur {base_url}. Vérifie le modèle, le port et l'URL."
        ) from exc

    try:
        json_payload = json.loads(response_data)
    except json.JSONDecodeError as exc:
        raise VaquilaError("Réponse API invalide: JSON illisible.") from exc

    choices = json_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VaquilaError("Réponse API invalide: champ `choices` manquant.")

    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    if not isinstance(message, dict):
        raise VaquilaError("Réponse API invalide: champ `message` manquant.")

    content = message.get("content")
    if isinstance(content, str):
        return _sanitize_model_output(content)

    if isinstance(content, list):
        text_parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                text_parts.append(chunk["text"])
        if text_parts:
            return _sanitize_model_output("".join(text_parts))

    raise VaquilaError("Réponse API invalide: contenu assistant vide.")
