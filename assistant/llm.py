"""Provider-agnostic chat client with tool calling.

Groq, OpenAI and Gemini all expose the OpenAI chat-completions protocol, so one class serves
them. Anthropic has its own message format and gets an adapter. Only `requests` is used, so no
vendor SDK has to install cleanly. Internally every message is OpenAI-shaped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

import requests
from django.conf import settings

log = logging.getLogger("assistant")

ORDER = ["groq", "gemini", "openai", "anthropic"]
KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
OPENAI_URLS = {"groq": "https://api.groq.com/openai/v1/chat/completions",
               "openai": "https://api.openai.com/v1/chat/completions",
               "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


# Groq models to prefer, best first, when the configured one is unavailable to the API key.
# (llama-3.3-70b-versatile and llama-3.1-8b-instant were shut down on 2026-08-16.)
GROQ_MODEL_PREFERENCE = ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b",
                         "meta-llama/llama-4-scout-17b-16e-instruct"]


class LLMError(Exception):
    def __init__(self, message, retry_after: float | None = None, retryable: bool = True, model_missing: bool = False):
        super().__init__(message)
        self.retry_after, self.retryable, self.model_missing = retry_after, retryable, model_missing


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""

    def as_message(self) -> dict:
        msg = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in self.tool_calls]
        return msg


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


class OpenAICompatible:
    def __init__(self, name, api_key, model):
        self.name, self.api_key, self.model, self.url = name, api_key, model, OPENAI_URLS[name]
        self._resolved = False

    def list_models(self) -> list[str]:
        """IDs of the models this API key can use (GET .../models)."""
        url = self.url.rsplit("/chat/completions", 1)[0] + "/models"
        try:
            r = requests.get(url, timeout=15, headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except (requests.RequestException, ValueError, KeyError):
            return []

    def resolve_model(self) -> bool:
        """The configured model is not available to this key: switch to the best one that is.
        Tried once per client. Only Groq has a preference list."""
        if self._resolved or self.name != "groq":
            return False
        self._resolved = True
        available = set(self.list_models())
        for candidate in GROQ_MODEL_PREFERENCE:
            if candidate in available and candidate != self.model:
                log.warning("Model %s is not available to this key; switching to %s", self.model, candidate)
                self.model = candidate
                return True
        log.error("Model %s unavailable and none of the preferred fallbacks are enabled. Available: %s",
                  self.model, sorted(available))
        return False

    OPTIONAL_PARAMS = ("parallel_tool_calls", "reasoning_effort", "reasoning_format")

    def _body(self, messages, tools, tool_choice) -> dict:
        reasoning = "gpt-oss" in self.model or "qwen" in self.model
        # reasoning models spend output tokens on thinking first: leave room for the answer
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "max_tokens": 2500 if reasoning else 700}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
            if self.name == "openai":
                body["parallel_tool_calls"] = False
        if self.name == "groq":
            if "gpt-oss" in self.model:
                body["reasoning_effort"] = "low"
            elif "qwen" in self.model:
                body["reasoning_format"] = "hidden"
        return body

    def chat(self, messages, tools, tool_choice=None, timeout=25.0) -> LLMResponse:
        body = self._body(messages, tools, tool_choice)
        for _ in range(3):
            try:
                r = requests.post(self.url, json=body, timeout=timeout,
                                  headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            except requests.RequestException as e:
                raise LLMError(f"{self.name}: network error {e.__class__.__name__}")
            rejected = [k for k in self.OPTIONAL_PARAMS if r.status_code == 400 and k in body and k in r.text]
            if not rejected:
                break
            for k in rejected:                      # a model that doesn't know an optional knob: drop it, retry
                body.pop(k)
        if r.status_code == 429:
            raise LLMError(f"{self.name}/{self.model}: rate limited", retry_after=_retry_after(r))
        if r.status_code >= 500:
            raise LLMError(f"{self.name}/{self.model}: server error {r.status_code}")
        if r.status_code == 404 and "model_not_found" in r.text:
            raise LLMError(f"{self.name}/{self.model}: model not available to this API key "
                           "(retired, or not enabled for the project): HTTP 404", retryable=False, model_missing=True)
        if r.status_code >= 400:
            # Models sometimes emit a malformed tool call -> 400 "tool_use_failed": a retry usually helps.
            retry = "tool_use_failed" in r.text or "failed to call a function" in r.text
            raise LLMError(f"{self.name}/{self.model}: HTTP {r.status_code} {r.text[:200]}", retryable=retry)
        try:
            msg = r.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError):
            raise LLMError(f"{self.name}/{self.model}: unreadable response")
        calls = [ToolCall(c.get("id") or f"call_{i}", c["function"]["name"], _parse_args(c["function"].get("arguments")))
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        content = _THINK.sub("", msg.get("content") or "").strip()
        return LLMResponse(content, calls, self.name)


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key, model):
        self.api_key, self.model = api_key, model

    @staticmethod
    def _convert(messages):
        system, out = "", []
        for m in messages:
            role = m["role"]
            if role == "system":
                system += m["content"] + "\n"
            elif role == "user":
                out.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for c in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                                   "input": _parse_args(c["function"]["arguments"])})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return system.strip(), out

    def chat(self, messages, tools, tool_choice=None, timeout=25.0) -> LLMResponse:
        system, msgs = self._convert(messages)
        body = {"model": self.model, "system": system, "messages": msgs, "max_tokens": 700, "temperature": 0}
        if tools:
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                              "input_schema": t["function"]["parameters"]} for t in tools]
            if tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
        try:
            r = requests.post(ANTHROPIC_URL, json=body, timeout=timeout, headers={
                "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        except requests.RequestException as e:
            raise LLMError(f"anthropic: network error {e.__class__.__name__}")
        if r.status_code in (429, 529):
            raise LLMError("anthropic: rate limited/overloaded", retry_after=_retry_after(r))
        if r.status_code >= 500:
            raise LLMError(f"anthropic: server error {r.status_code}")
        if r.status_code >= 400:
            raise LLMError(f"anthropic: HTTP {r.status_code} {r.text[:200]}", retryable=False)
        blocks = r.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        calls = [ToolCall(b["id"], b["name"], _parse_args(b.get("input"))) for b in blocks if b.get("type") == "tool_use"]
        return LLMResponse(text, calls, "anthropic")


def _retry_after(r) -> float | None:
    try:
        return float(r.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def configured_providers() -> list:
    """Clients for every provider that has a key; the preferred one first."""
    clients = {}
    for name in ORDER:
        key = os.environ.get(KEY_ENV[name], "").strip()
        if key:
            model = settings.LLM_MODELS[name]
            clients[name] = AnthropicClient(key, model) if name == "anthropic" else OpenAICompatible(name, key, model)
    preferred = settings.LLM_PROVIDER
    order = ([preferred] if preferred in clients else []) + [n for n in ORDER if n in clients and n != preferred]
    out = []
    for n in order:
        out.append(clients[n])
        fb = settings.GROQ_FALLBACK_MODEL
        if n == "groq" and fb and fb != settings.LLM_MODELS["groq"]:   # same key, different model
            out.append(OpenAICompatible("groq", clients[n].api_key, fb))
    return out


class Router:
    """Tries providers in order. Rate limits get one short wait; other failures fall through."""

    def __init__(self, providers, deadline: float):
        self.providers, self.deadline = providers, deadline
        self.used: str | None = None

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def chat(self, messages, tools, tool_choice=None) -> LLMResponse:
        if not self.providers:
            raise LLMError("No LLM provider configured (set GROQ_API_KEY or another provider key).", retryable=False)
        last = None
        for client in self.providers:
            for attempt in range(2):
                if self.remaining() < 4:
                    raise LLMError("Time budget exhausted.", retryable=False)
                try:
                    resp = client.chat(messages, tools, tool_choice, timeout=min(25.0, self.remaining() - 1))
                    self.used = client.name
                    return resp
                except LLMError as e:
                    last = e
                    log.warning("LLM call failed (%s), attempt %d: %s", client.name, attempt + 1, e)
                    if e.model_missing and getattr(client, "resolve_model", lambda: False)():
                        continue                        # retry immediately with the model that is available
                    if not e.retryable:
                        break
                    if e.retry_after is not None:
                        if e.retry_after > min(8.0, self.remaining() - 6):
                            break                       # waiting would eat the budget: try the next provider
                        time.sleep(e.retry_after + 0.2)
                    elif attempt == 0:
                        time.sleep(0.4)
        raise last or LLMError("All providers failed.")