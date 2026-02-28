"""LLM provider abstraction for the QRL platform."""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMProvider(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a response given a prompt and optional system message."""


@dataclass
class OllamaProvider(LLMProvider):
    """Local Ollama provider.

    Requires Ollama running at localhost:11434.
    Models available: deepseek-coder-v2:16b, marco:latest
    """

    model: str = "marco:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1  # low temperature for deterministic code gen
    timeout: int = 120

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["message"]["content"].strip()


@dataclass
class TogetherAIProvider(LLMProvider):
    """Together.ai cloud provider (~$0.20-0.27/M tokens).

    Set TOGETHER_API_KEY environment variable.
    """

    model: str = "deepseek-ai/DeepSeek-V3"
    api_key: str = ""
    temperature: float = 0.1
    timeout: int = 60

    def __post_init__(self) -> None:
        import os
        if not self.api_key:
            self.api_key = os.environ.get("TOGETHER_API_KEY", "")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set")

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://api.together.xyz/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip()
