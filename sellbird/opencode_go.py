"""OpenCode Go as a Sellbird brain: one text-only chat completion per call.

Sellbird's prompts carry untrusted text (scraped pages, prospect replies) and
its output reaches third parties as email, so the model gets nothing to act
with. Every request here is a single ``user`` message with no ``tools``,
``tool_choice`` or ``stream``; a reply that carries ``tool_calls`` is an
error. The only host ever called is OpenCode's own, fixed below, redirects are
not followed, and the API key is read at call time and never logged, put in an
error message, or echoed back.

OpenCode Go is documented as "designed for OpenCode and other coding agents".
Sellbird is not one, so it says what it is: ``User-Agent: sellbird/<version>``
and a stable ``x-opencode-session`` per Brain, as the service asks of clients.
It never imitates OpenCode or a coding agent (see docs/opencode-go.md).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version

import httpx
from dotenv import dotenv_values

from sellbird.paths import PROJECT_ROOT

logger = logging.getLogger("sellbird.opencode_go")

BASE_URL = "https://opencode.ai/inference/go/openai/v1"
MODELS_URL = "https://opencode.ai/inference/go/v1/models"
API_KEY_ENV = "OPENCODE_GO_API_KEY"
# Reasoning tokens count against max_tokens, so leave room for them.
MAX_TOKENS = 16000
RETRY_MAX_TOKENS = 32000

_LIMIT_WORDS = ("limit", "quota", "allowance")
_SETUP_WORDS = ("unavailable", "unknown model", "not found", "api key", "session")


class OpenCodeGoError(Exception):
    """An OpenCode Go call failed. ``kind`` drives retry and the operator message.

    Kinds mirror CodexError so Brain handles both alike: ``auth``, ``setup``,
    ``usage_limit``, ``transient``, ``paused`` and ``protocol``. Messages never
    contain prompts, replies, response bodies, headers or the key.
    """

    def __init__(self, message: str, kind: str = "transient", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass
class OpenCodeGoResult:
    text: str
    model: str = ""
    thread_id: str = ""  # the x-opencode-session used, like Codex's thread id
    duration_ms: int = 0
    usage: dict = field(default_factory=dict)
    rate_limits: dict | None = None


def _client_version() -> str:
    try:
        return version("sellbird")
    except PackageNotFoundError:
        return "0"


def _int(value) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


class OpenCodeGoClient:
    def __init__(
        self,
        model: str,
        reasoning_effort: str = "",
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._api_key = api_key
        self._transport = transport
        self._session = "sellbird-" + uuid.uuid4().hex[:16]

    # ── credentials ──

    def _key(self) -> str:
        """The API key, trimmed; blank counts as missing. Order: the value
        given at construction, the environment, then the project's .env."""
        raw = self._api_key
        if raw is None:
            raw = os.environ.get(API_KEY_ENV)
        if raw is None:
            raw = dotenv_values(PROJECT_ROOT / ".env").get(API_KEY_ENV)
        return (raw or "").strip()

    def _require_key(self) -> str:
        key = self._key()
        if not key:
            raise OpenCodeGoError(
                f"No OpenCode Go API key. Set {API_KEY_ENV} in .env "
                "(sellbird setup asks for it).", kind="setup")
        return key

    def fingerprint(self) -> tuple:
        """What a paused Brain re-checks: a new key or model un-pauses it.
        The key itself is only ever represented by a short one-way hash."""
        key = self._key()
        digest = sha256(key.encode()).hexdigest()[:12] if key else ""
        return ("opencode_go", digest, self.model)

    # ── transport ──

    def _headers(self, key: str) -> dict:
        return {
            "Authorization": f"Bearer {key}",
            "User-Agent": f"sellbird/{_client_version()}",
            "x-opencode-session": self._session,
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, url: str, key: str, timeout: float,
                       body: dict | None = None) -> dict:
        """One HTTP call → the parsed JSON object, or an OpenCodeGoError."""
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=timeout, follow_redirects=False
            ) as http:
                response = await http.request(
                    method, url, headers=self._headers(key), json=body)
        except httpx.TimeoutException:
            raise OpenCodeGoError("OpenCode Go call timed out.", "transient", True) from None
        except httpx.TransportError:
            raise OpenCodeGoError("Could not reach OpenCode Go.", "transient", True) from None
        raise_for_status(response)
        try:
            data = response.json()
        except ValueError:
            raise OpenCodeGoError(
                "OpenCode Go returned something that is not JSON.", "transient", True) from None
        if not isinstance(data, dict):
            raise OpenCodeGoError(
                "OpenCode Go returned an unexpected reply.", "transient", True)
        return data

    # ── a call ──

    async def run(self, prompt: str, timeout: float, skip_if=None) -> OpenCodeGoResult:
        if skip_if is not None:
            reason = skip_if()
            if reason:
                raise OpenCodeGoError(f"OpenCode Go call skipped: {reason}", "paused")
        key = self._require_key()
        started = time.monotonic()
        data = None
        for max_tokens in (MAX_TOKENS, RETRY_MAX_TOKENS):
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
            data = await self._request("POST", BASE_URL + "/chat/completions", key,
                                       timeout, body)
            if _finish_reason(data) != "length":
                break
        else:
            raise OpenCodeGoError(
                "OpenCode Go reply was cut off even with a larger token budget.",
                "transient", False)
        text = _parse_text(data)
        return OpenCodeGoResult(
            text=text,
            model=str(data.get("model") or self.model),
            thread_id=self._session,
            duration_ms=int((time.monotonic() - started) * 1000),
            usage=_parse_usage(data),
        )


def raise_for_status(response: httpx.Response) -> None:
    """Map an HTTP failure to an OpenCodeGoError. Only the status and a few
    keywords from the body are used; the body itself is never repeated."""
    status = response.status_code
    if status < 300:
        return
    if status < 400:
        raise OpenCodeGoError(
            f"OpenCode Go answered with an unexpected redirect ({status}).", "protocol")
    if status in (401, 403):
        raise OpenCodeGoError(
            "OpenCode Go rejected the API key. Check OPENCODE_GO_API_KEY and that your "
            "Go subscription is active.", "auth")
    if status >= 500:
        raise OpenCodeGoError(f"OpenCode Go server error ({status}).", "transient", True)
    text = response.text.lower()
    if status == 429 or any(word in text for word in _LIMIT_WORDS):
        raise OpenCodeGoError(
            "OpenCode Go usage limit reached for this plan.", "usage_limit")
    if any(word in text for word in _SETUP_WORDS):
        raise OpenCodeGoError(
            "OpenCode Go can't use this model or request (model unavailable or "
            "not allowed). Check brain.model.", "setup")
    raise OpenCodeGoError(f"OpenCode Go rejected the request ({status}).", "protocol")


def _choice(data: dict) -> dict:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    raise OpenCodeGoError("OpenCode Go reply had no choices.", "transient", True)


def _finish_reason(data: dict) -> str:
    return str(_choice(data).get("finish_reason") or "")


def _parse_text(data: dict) -> str:
    """The model's answer: ``content`` only. Reasoning text is never returned,
    and a reply that asks for a tool call is refused."""
    choice = _choice(data)
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
        raise OpenCodeGoError(
            "OpenCode Go reply asked for a tool call; Sellbird sends no tools.", "protocol")
    finish = choice.get("finish_reason")
    if finish not in ("stop", None):
        raise OpenCodeGoError(
            f"OpenCode Go reply ended unexpectedly ({finish}).", "protocol")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OpenCodeGoError(
            "OpenCode Go reply had no answer text.", "transient", True)
    return content.strip()


def _parse_usage(data: dict) -> dict:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details")
    cached = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    extra = usage.get("opencode") if isinstance(usage.get("opencode"), dict) else {}
    tokens = extra.get("tokens") if isinstance(extra.get("tokens"), dict) else {}
    allowance = extra.get("allowance")
    return {
        "input_tokens": max(_int(usage.get("prompt_tokens")) - cached, 0),
        "output_tokens": _int(usage.get("completion_tokens")),
        "cache_read_tokens": cached,
        "cache_creation_tokens": _int(tokens.get("cache_write_5m"))
        + _int(tokens.get("cache_write_1h")),
        "cost_usd": float(allowance) if isinstance(allowance, (int, float)) else 0.0,
    }
