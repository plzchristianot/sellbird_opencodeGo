"""The OpenCode Go adapter: request shape, error mapping and the text-only guards."""

import asyncio
import json
import logging

import httpx
import pytest

from sellbird.opencode_go import (
    BASE_URL,
    RETRY_MAX_TOKENS,
    OpenCodeGoError,
)
from tests.fake_opencode_go import TEST_KEY, client_with, completion, ok


@pytest.mark.asyncio
async def test_run_returns_content_only_and_usage():
    payload = completion(content='{"ok":true}', reasoning="thinking out loud")
    result = await client_with(lambda r: ok(payload)).run("hi", timeout=30)
    assert result.text == '{"ok":true}'
    assert "thinking" not in result.text
    assert result.usage["cost_usd"] == 0.00001
    assert result.usage["output_tokens"] == 19
    assert result.model == "deepseek-v4.1-flash"


@pytest.mark.asyncio
async def test_request_shape():
    seen = []

    def handler(request):
        seen.append(request)
        return ok()

    client = client_with(handler, reasoning_effort="low")
    await client.run("first", timeout=30)
    await client.run("second", timeout=30)

    first, second = seen
    assert str(first.url) == BASE_URL + "/chat/completions"
    assert first.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert first.headers["user-agent"].startswith("sellbird/")
    assert first.headers["x-opencode-session"]
    assert first.headers["x-opencode-session"] == second.headers["x-opencode-session"]
    body = json.loads(first.content)
    assert body["messages"] == [{"role": "user", "content": "first"}]
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["reasoning_effort"] == "low"
    for forbidden in ("tools", "tool_choice", "stream"):
        assert forbidden not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("response, kind, retryable", [
    (httpx.Response(401, json={"error": {"message": "bad key"}}), "auth", False),
    (httpx.Response(403, json={"error": {"message": "no access"}}), "auth", False),
    (httpx.Response(429, json={"error": {"message": "slow down"}}), "usage_limit", False),
    (httpx.Response(400, json={"error": {"message": "Upstream request failed: Model is unavailable."}}),
     "setup", False),
    (httpx.Response(400, json={"error": {"message": "Monthly usage limit reached"}}),
     "usage_limit", False),
    (httpx.Response(500, json={"error": {"message": "boom"}}), "transient", True),
    (httpx.Response(502, text="bad gateway"), "transient", True),
])
async def test_error_mapping(response, kind, retryable):
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: response).run("hi", timeout=30)
    assert caught.value.kind == kind
    assert caught.value.retryable is retryable


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [httpx.ReadTimeout("t"), httpx.ConnectError("c")])
async def test_network_failures_are_transient(exc):
    def handler(request):
        raise exc

    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(handler).run("hi", timeout=30)
    assert (caught.value.kind, caught.value.retryable) == ("transient", True)


@pytest.mark.asyncio
async def test_tool_calls_rejected():
    payload = completion(content="", finish="tool_calls",
                         tool_calls=[{"id": "c1", "type": "function",
                                      "function": {"name": "x", "arguments": "{}"}}])
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: ok(payload)).run("hi", timeout=30)
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_length_finish_retries_once_then_fails():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return ok(completion(content="cut off", finish="length"))

    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(handler).run("hi", timeout=30)
    assert caught.value.kind == "transient"
    assert len(bodies) == 2
    assert bodies[1]["max_tokens"] == RETRY_MAX_TOKENS
    assert bodies[1]["max_tokens"] > bodies[0]["max_tokens"]


@pytest.mark.asyncio
async def test_length_then_stop_succeeds():
    replies = [completion(content="", finish="length"), completion(content="done")]
    result = await client_with(lambda r: ok(replies.pop(0))).run("hi", timeout=30)
    assert result.text == "done"


@pytest.mark.asyncio
async def test_redirect_not_followed():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://example.com/steal"})

    with pytest.raises(OpenCodeGoError):
        await client_with(handler).run("hi", timeout=30)
    assert calls == [BASE_URL + "/chat/completions"]


@pytest.mark.asyncio
async def test_reasoning_only_reply_is_not_an_answer():
    payload = completion(content="", reasoning="the answer is 42")
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: ok(payload)).run("hi", timeout=30)
    assert caught.value.kind == "transient"
    assert "42" not in str(caught.value)


@pytest.mark.asyncio
async def test_key_never_leaks(caplog):
    caplog.set_level(logging.DEBUG)
    leak = httpx.Response(500, text=f"echo Authorization: Bearer {TEST_KEY}")
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: leak).run("prompt text", timeout=30)
    shown = str(caught.value) + caplog.text
    assert TEST_KEY not in shown
    assert "Bearer" not in shown
    assert "prompt text" not in shown


@pytest.mark.asyncio
async def test_padded_key_is_stripped():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        return ok()

    await client_with(handler, api_key=f" {TEST_KEY}\n").run("hi", timeout=30)
    assert seen["auth"] == f"Bearer {TEST_KEY}"


@pytest.mark.asyncio
async def test_blank_key_is_missing():
    called = []
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: called.append(r), api_key="   ").run("hi", timeout=30)
    assert caught.value.kind == "setup"
    assert "OPENCODE_GO_API_KEY" in str(caught.value)
    assert called == []


@pytest.mark.asyncio
async def test_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "  env-key-not-real ")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        return ok()

    from sellbird.opencode_go import OpenCodeGoClient
    client = OpenCodeGoClient("deepseek-v4.1-flash", transport=httpx.MockTransport(handler))
    await client.run("hi", timeout=30)
    assert seen["auth"] == "Bearer env-key-not-real"


@pytest.mark.asyncio
async def test_non_json_200_is_transient():
    page = httpx.Response(200, text="<html>captive portal</html>")
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: page).run("hi", timeout=30)
    assert (caught.value.kind, caught.value.retryable) == ("transient", True)


@pytest.mark.asyncio
async def test_skip_if_pauses_without_a_request():
    called = []
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: called.append(r)).run(
            "hi", timeout=30, skip_if=lambda: "usage limit reached")
    assert caught.value.kind == "paused"
    assert called == []


@pytest.mark.asyncio
async def test_cancellation_propagates():
    def handler(request):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await client_with(handler).run("hi", timeout=30)
