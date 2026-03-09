"""HTTP call services for the OpenAI-compatible vLLM API."""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from vaquila.exceptions import VaquilaError


def _sanitize_model_output(text: str) -> str:
    """Remove optional reasoning tags for clean user output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned if cleaned else text.strip()


def _candidate_base_urls(base_url: str) -> list[str]:
    """Return candidate inference base URLs for host and Docker Desktop contexts."""
    candidates: list[str] = []

    def _push(url: str) -> None:
        if url not in candidates:
            candidates.append(url)

    normalized = base_url.rstrip("/")
    _push(normalized)

    parsed = urlsplit(normalized)
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "0.0.0.0"}:
        netloc = "host.docker.internal"
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        fallback = urlunsplit((parsed.scheme or "http", netloc, parsed.path, parsed.query, parsed.fragment))
        _push(fallback.rstrip("/"))

    return candidates


def run_inference(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
    """Run a chat completion request and return the assistant response."""
    if max_tokens <= 0:
        raise VaquilaError("max_tokens must be greater than 0.")
    if timeout_seconds <= 0:
        raise VaquilaError("timeout must be greater than 0.")

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode("utf-8")

    last_error: Exception | None = None
    last_base_url = base_url.rstrip("/")
    candidate_base_urls = _candidate_base_urls(base_url)
    for index, candidate_base_url in enumerate(candidate_base_urls):
        endpoint = f"{candidate_base_url.rstrip('/')}/v1/chat/completions"
        request = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_data = response.read().decode("utf-8")
            last_base_url = candidate_base_url
            break
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            # Some legacy/base models (e.g. GPT-2) do not expose a chat template.
            # In that case, retry once with the completions endpoint.
            if exc.code == 400 and "chat template" in details.lower():
                completion_payload = {
                    "model": model_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                completion_body = json.dumps(completion_payload).encode("utf-8")
                completion_endpoint = f"{candidate_base_url.rstrip('/')}/v1/completions"
                completion_request = Request(
                    completion_endpoint,
                    data=completion_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen(completion_request, timeout=timeout_seconds) as response:
                        response_data = response.read().decode("utf-8")
                    last_base_url = candidate_base_url
                    break
                except HTTPError as completion_exc:
                    completion_details = completion_exc.read().decode("utf-8", errors="replace")
                    last_error = VaquilaError(
                        "vLLM API request failed on both chat and completion endpoints "
                        f"at {candidate_base_url} (chat={exc.code}, completion={completion_exc.code}): "
                        f"{completion_details}"
                    )
                    if index < len(candidate_base_urls) - 1:
                        continue
                    raise last_error from completion_exc
                except URLError as completion_exc:
                    last_error = VaquilaError(
                        f"Unable to reach vLLM completions API at {candidate_base_url}. "
                        "Verify model, port, and URL."
                    )
                    if index < len(candidate_base_urls) - 1:
                        continue
                    raise last_error from completion_exc

            last_error = VaquilaError(
                f"vLLM API request failed ({exc.code}) at {candidate_base_url}: {details}"
            )
            if index < len(candidate_base_urls) - 1:
                continue
            raise last_error from exc
        except URLError as exc:
            last_error = VaquilaError(
                f"Unable to reach vLLM API at {candidate_base_url}. Verify model, port, and URL."
            )
            if index < len(candidate_base_urls) - 1:
                continue
            raise last_error from exc
    else:
        if isinstance(last_error, VaquilaError):
            raise last_error
        raise VaquilaError(f"Unable to reach vLLM API at {last_base_url}.")

    try:
        json_payload = json.loads(response_data)
    except json.JSONDecodeError as exc:
        raise VaquilaError("Invalid API response: unreadable JSON.") from exc

    choices = json_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VaquilaError("Invalid API response: missing `choices` field.")

    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    if not isinstance(message, dict):
        text_value = first_choice.get("text") if isinstance(first_choice, dict) else None
        if isinstance(text_value, str):
            return _sanitize_model_output(text_value)
        raise VaquilaError("Invalid API response: missing `message` field.")

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

    raise VaquilaError("Invalid API response: assistant content is empty.")
