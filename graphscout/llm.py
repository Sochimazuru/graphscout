"""Thin client for a local Ollama server that always returns parsed JSON."""
from __future__ import annotations

import json
import re

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


class ModelNotFound(LLMError):
    pass


def _close_brackets(prefix: str) -> str | None:
    """Return `prefix` with its still-open [ and { closed, or None if it ends inside a string."""
    stack, in_str, esc = [], False, False
    for ch in prefix:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        return None
    return prefix + "".join(reversed(stack))


def repair_truncated(text: str, max_attempts: int = 200) -> dict | None:
    """Rescue a JSON answer that was cut off mid-way.

    Walks back through the text to the last complete object or list item,
    closes any open brackets, and returns whatever parses. Facts after the cut are lost;
    facts before it are kept exactly as the model wrote them.
    """
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    # Only cut right after a closing bracket, so a half-written item is dropped, never kept.
    cuts = [i + 1 for i, ch in enumerate(body) if ch in "}]"]
    for cut in sorted(cuts, reverse=True)[:max_attempts]:
        candidate = _close_brackets(body[:cut])
        if candidate is None:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    repaired = repair_truncated(text)
    if repaired is not None:
        return repaired
    raise json.JSONDecodeError("unrecoverable JSON from model", text[:200], 0)


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
        wanted = self.cfg.model if ":" in self.cfg.model else self.cfg.model + ":latest"
        if wanted not in names:
            installed = ", ".join(sorted(names)) or "none"
            raise LLMError(
                f"Model {self.cfg.model} is not installed. Run `ollama pull {self.cfg.model}`, "
                f"or use one you already have with --model. Installed: {installed}"
            )

    def json(self, system: str, user: str, retries: int = 2) -> dict:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": self.cfg.max_output_tokens},
        }
        # Only network problems are retried. With temperature 0 the same prompt gives the
        # same answer, so retrying after bad JSON would just repeat the same mistake.
        last_err: Exception | None = None
        for _ in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.cfg.ollama_url}/api/chat", json=payload, timeout=self.cfg.llm_timeout
                )
            except requests.RequestException as exc:
                last_err = exc
                continue
            if resp.status_code >= 500:  # Ollama hiccup (often memory); worth one more try
                last_err = requests.HTTPError(f"{resp.status_code} from Ollama: {resp.text[:200]}")
                continue
            if resp.status_code == 404:
                raise ModelNotFound(
                    f"Ollama says model {self.cfg.model} was not found. Run `ollama pull {self.cfg.model}`."
                )
            try:
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                raise LLMError(f"Bad response from Ollama: {exc}") from exc
            try:
                return _parse_json(content)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Model returned JSON that could not be repaired: {exc}") from exc
        raise LLMError(f"Ollama failed after {retries + 1} attempts: {last_err}")
