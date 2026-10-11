"""Thin client for a local Ollama server that always returns parsed JSON."""
from __future__ import annotations

import json
import re

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class OllamaLLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def check(self) -> None:
        """Fail early, with a clear message, if Ollama or the model is missing."""
        try:
            tags = requests.get(f"{self.cfg.ollama_url}/api/tags", timeout=10).json()
        except requests.RequestException as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.cfg.ollama_url}. Is `ollama serve` running?"
            ) from exc
        names = {m.get("name", "") for m in tags.get("models", [])}
        base = self.cfg.model.split(":")[0]
        if self.cfg.model not in names and not any(n.split(":")[0] == base for n in names):
            raise LLMError(f"Model {self.cfg.model} not found. Run `ollama pull {self.cfg.model}`.")

    def json(self, system: str, user: str, retries: int = 2) -> dict:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        }
        last_err: Exception | None = None
        for _ in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.cfg.ollama_url}/api/chat", json=payload, timeout=self.cfg.llm_timeout
                )
                resp.raise_for_status()
                return _parse_json(resp.json()["message"]["content"])
            except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
                last_err = exc
        raise LLMError(f"LLM call failed after {retries + 1} attempts: {last_err}")
