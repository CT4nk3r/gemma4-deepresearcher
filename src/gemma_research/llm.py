from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .config import ModelConfig


class LLMError(RuntimeError):
    """Raised when a model endpoint fails."""


class ChatClient(Protocol):
    provider: str

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant content for a chat completion."""


@dataclass
class OpenAICompatibleClient:
    config: ModelConfig
    provider: str = "openai-compatible"

    def complete(self, messages: list[dict[str, str]]) -> str:
        base_url = _openai_base_url(self.config.base_url)
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"{self.provider} HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider} response missing message content") from exc


@dataclass
class OllamaClient:
    config: ModelConfig
    provider: str = "ollama"

    def complete(self, messages: list[dict[str, str]]) -> str:
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"Ollama HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama response missing message content") from exc


@dataclass
class OfflineClient:
    provider: str = "offline"

    def complete(self, messages: list[dict[str, str]]) -> str:
        last_user = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )
        return (
            "Offline deterministic provider. Configure LM Studio or Ollama for model "
            f"generation. Last request summary: {last_user[:400]}"
        )


def create_client(config: ModelConfig) -> ChatClient:
    provider = config.provider.lower()
    if provider == "offline":
        return OfflineClient()
    if provider == "lmstudio":
        model_config = _with_defaults(config, base_url="http://localhost:1234/v1")
        return OpenAICompatibleClient(model_config, provider="lmstudio")
    if provider == "openai":
        model_config = _with_defaults(config, base_url="https://api.openai.com/v1")
        return OpenAICompatibleClient(model_config, provider="openai")
    if provider == "openai-compatible":
        if not config.base_url:
            raise LLMError("openai-compatible provider requires model.base_url")
        return OpenAICompatibleClient(config, provider="openai-compatible")
    if provider == "ollama":
        model_config = _with_defaults(config, base_url="http://localhost:11434")
        return OllamaClient(model_config)
    raise LLMError(f"Unsupported model provider: {config.provider}")


def _with_defaults(config: ModelConfig, *, base_url: str) -> ModelConfig:
    return ModelConfig(
        provider=config.provider,
        model=config.model,
        base_url=config.base_url or base_url,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def _openai_base_url(base_url: str) -> str:
    if not base_url:
        base_url = "http://localhost:1234/v1"
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url
