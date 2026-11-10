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
CATALOG_TTL = 300.0
CATALOG_FAIL_TTL = 60.0

# Plan-allowance wording. A bare "limit" is not enough: "context limit" and
# "output limit" describe one request, not the plan, and must not stop every call.
_PLAN_PHRASES = ("usage limit", "quota", "allowance", "limit reached", "limit exceeded",
                 "monthly limit", "weekly limit")
_REQUEST_WORDS = ("context", "token")
_TOO_LARGE_WORDS = ("context", "max_tokens", "too long", "too large")
_SETUP_WORDS = ("unavailable", "unknown model", "not found", "api key", "session")
# A Retry-After this long means the plan is spent, not a momentary rate limit.
_LONG_RETRY_SECONDS = 600


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
    thread_id: str = ""  # unique per call (the session header plus a suffix)
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
        # The model list (or why it couldn't be read), cached briefly and only
        # for the key it was read with.
        self._catalog: dict | None = None
        self._catalog_error: OpenCodeGoError | None = None
        self._catalog_at = 0.0
        self._catalog_digest = ""

    # ── credentials ──

    def _key(self) -> str:
        """The API key, trimmed; blank counts as missing. Order: the value
        given at construction, then the project's .env, then the environment.
        The file comes before the environment because ``sellbird run`` loads
        .env into the environment once at startup: a key fixed in the file
        afterwards must win over that stale copy. A blank line in .env (as in
        .env.example) falls through to the environment."""
        if self._api_key is not None:
            return self._api_key.strip()
        from_file = (dotenv_values(PROJECT_ROOT / ".env").get(API_KEY_ENV) or "").strip()
        return from_file or (os.environ.get(API_KEY_ENV) or "").strip()

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

    # ── the model catalog ──

    def _key_digest(self) -> str:
        key = self._key()
        return sha256(key.encode()).hexdigest()[:12] if key else ""

    async def list_models(self) -> dict[str, dict]:
        """id -> {"protocol", "efforts", "output_limit"} from the live catalog.

        Cached for CATALOG_TTL (CATALOG_FAIL_TTL for a failure, so a broken
        service isn't asked every heartbeat), and only for the key it was read
        with, so a new key is tried at once."""
        key = self._require_key()
        digest = self._key_digest()
        if digest != self._catalog_digest:
            self._catalog = self._catalog_error = None
            self._catalog_digest = digest
        age = time.monotonic() - self._catalog_at
        if self._catalog is not None and age < CATALOG_TTL:
            return self._catalog
        if self._catalog_error is not None and age < CATALOG_FAIL_TTL:
            raise self._catalog_error
        try:
            models = _parse_catalog(await self._request("GET", MODELS_URL, key, 30.0))
        except OpenCodeGoError as e:
            self._catalog, self._catalog_error = None, e
            self._catalog_at = time.monotonic()
            raise
        self._catalog, self._catalog_error = models, None
        self._catalog_at = time.monotonic()
        return models

    async def preflight(self) -> None:
        """Can a call be made right now? Reads the model list with the key and
        checks the configured model and effort against it. Sends no prompt and
        spends no allowance."""
        models = await self.list_models()
        usable = ", ".join(supported_models(models)) or "none listed"
        entry = models.get(self.model)
        if entry is None:
            raise OpenCodeGoError(
                f"Model '{self.model}' is not in your OpenCode Go catalog. "
                f"Models Sellbird can use: {usable}.", "setup")
        if entry["protocol"] != "chat-completions":
            raise OpenCodeGoError(
                f"Model '{self.model}' uses the '{entry['protocol'] or 'unknown'}' protocol; "
                f"Sellbird supports chat-completions models only: {usable}.", "setup")
        if self.reasoning_effort and self.reasoning_effort not in entry["efforts"]:
            raise OpenCodeGoError(
                f"Model '{self.model}' doesn't offer reasoning_effort "
                f"'{self.reasoning_effort}'. Valid: "
                f"{', '.join(entry['efforts']) or 'none (leave it blank)'}.", "setup")

    def _cap(self, tokens: int) -> int:
        """Never ask for more output than the model's catalog entry allows."""
        entry = (self._catalog or {}).get(self.model) or {}
        limit = entry.get("output_limit")
        return min(tokens, limit) if limit else tokens

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
                "max_tokens": self._cap(max_tokens),
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
            # Usage views count DISTINCT session_id as calls, so each call gets
            # its own id even though the x-opencode-session header is stable.
            thread_id=f"{self._session}:{uuid.uuid4().hex[:8]}",
            duration_ms=int((time.monotonic() - started) * 1000),
            usage=_parse_usage(data),
        )


def raise_for_status(response: httpx.Response) -> None:
    """Map an HTTP failure to an OpenCodeGoError. Only the status, one header
    and a few keywords from the body are used; the body is never repeated."""
    status = response.status_code
    if status < 300:
        return
    if status < 400:
        raise OpenCodeGoError(
            f"OpenCode Go answered with an unexpected redirect ({status}).", "protocol")
    if status >= 500:
        raise OpenCodeGoError(f"OpenCode Go server error ({status}).", "transient", True)
    text = response.text.lower()
    plan_spent = (any(p in text for p in _PLAN_PHRASES)
                  and not any(w in text for w in _REQUEST_WORDS))
    if plan_spent:
        raise OpenCodeGoError(
            "OpenCode Go usage limit reached for this plan.", "usage_limit")
    if status in (401, 403):
        raise OpenCodeGoError(
            "OpenCode Go rejected the API key. Check OPENCODE_GO_API_KEY and that your "
            "Go subscription is active.", "auth")
    if status == 429:
        if _retry_after_seconds(response) >= _LONG_RETRY_SECONDS:
            raise OpenCodeGoError(
                "OpenCode Go usage limit reached for this plan.", "usage_limit")
        raise OpenCodeGoError(
            "OpenCode Go is rate limiting requests; trying again shortly.", "transient", True)
    if any(word in text for word in _TOO_LARGE_WORDS):
        raise OpenCodeGoError(
            "OpenCode Go refused the request as too large for this model "
            "(prompt or max_tokens). Nothing is paused.", "protocol")
    if any(word in text for word in _SETUP_WORDS):
        raise OpenCodeGoError(
            "OpenCode Go can't use this model or request (model unavailable or "
            "not allowed). Check brain.model.", "setup")
    raise OpenCodeGoError(f"OpenCode Go rejected the request ({status}).", "protocol")


def _retry_after_seconds(response: httpx.Response) -> float:
    """The Retry-After header in seconds; 0 when absent or not a number."""
    try:
        return float(response.headers.get("retry-after", "0"))
    except ValueError:
        return 0.0


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


def _parse_catalog(data: dict) -> dict[str, dict]:
    """Read the /models reply defensively: entries missing pieces are kept as
    unusable (empty protocol), never a KeyError."""
    entries = data.get("data")
    if not isinstance(entries, list):
        raise OpenCodeGoError("OpenCode Go model list was unreadable.", "transient", True)
    models: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        api = entry.get("api")
        protocol = next(iter(api), "") if isinstance(api, dict) and api else ""
        caps = entry.get("capabilities")
        reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
        efforts = reasoning.get("effort") if isinstance(reasoning, dict) else None
        limit = entry.get("limit")
        output = limit.get("output") if isinstance(limit, dict) else None
        models[entry["id"]] = {
            "protocol": protocol if isinstance(protocol, str) else "",
            "efforts": [e for e in efforts if isinstance(e, str)]
            if isinstance(efforts, list) else [],
            "output_limit": output if isinstance(output, int) and output > 0 else None,
        }
    return models


def supported_models(catalog: dict[str, dict]) -> list[str]:
    """Ids Sellbird can call in v1: those on the chat-completions protocol."""
    return sorted(k for k, v in catalog.items() if v.get("protocol") == "chat-completions")


class OpenCodeGoQuotaClient:
    """Same interface as the Claude and Codex quota gauges. OpenCode Go has no
    documented balance endpoint, so there is nothing to read: the budget gate
    falls back to Sellbird's own call counter and the dashboard says so."""

    def __init__(self, client: OpenCodeGoClient | None = None):
        self.client = client

    def invalidate(self) -> None:
        """Nothing cached to forget."""

    async def get_utilization(self) -> None:
        return None
