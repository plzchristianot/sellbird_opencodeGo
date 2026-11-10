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
    (httpx.Response(429, json={"error": {"message": "Monthly usage limit reached"}}),
     "usage_limit", False),
    (httpx.Response(429, json={"error": {"message": "Too many requests, retry in 2s"}}),
     "transient", True),
    (httpx.Response(429, headers={"retry-after": "3600"}, json={"error": {"message": "slow down"}}),
     "usage_limit", False),
    (httpx.Response(403, json={"error": {"message": "Your plan's usage limit was reached"}}),
     "usage_limit", False),
    (httpx.Response(400, json={"error": {"message": "Prompt exceeds the model's context limit"}}),
     "protocol", False),
    (httpx.Response(400, json={"error": {"message": "max_tokens is too large; output limit is 8000"}}),
     "protocol", False),
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


# ── catalog, preflight, quota ──

from sellbird.opencode_go import MODELS_URL, OpenCodeGoQuotaClient, supported_models  # noqa: E402
from tests.fake_opencode_go import DEFAULT_CATALOG, catalog  # noqa: E402


def _catalog_handler(payload=None, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload if payload is not None else catalog(DEFAULT_CATALOG))
    return handler


@pytest.mark.asyncio
async def test_preflight_success_sends_no_prompt():
    seen = []
    await client_with(_catalog_handler(seen=seen)).preflight()
    assert [(r.method, str(r.url)) for r in seen] == [("GET", MODELS_URL)]
    assert seen[0].headers["authorization"] == f"Bearer {TEST_KEY}"
    assert seen[0].headers["user-agent"].startswith("sellbird/")


@pytest.mark.asyncio
async def test_preflight_missing_key_makes_no_request():
    seen = []
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(_catalog_handler(seen=seen), api_key="").preflight()
    assert caught.value.kind == "setup"
    assert seen == []


@pytest.mark.asyncio
async def test_preflight_bad_key_is_auth():
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(lambda r: httpx.Response(401, json={})).preflight()
    assert caught.value.kind == "auth"


@pytest.mark.asyncio
async def test_preflight_unknown_model_lists_supported_ids():
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(_catalog_handler(), model="no-such-model").preflight()
    assert caught.value.kind == "setup"
    assert "deepseek-v4.1-flash" in str(caught.value)
    assert "grok-4.7" not in str(caught.value)  # not usable in v1, so not offered


@pytest.mark.asyncio
async def test_preflight_wrong_protocol():
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(_catalog_handler(), model="grok-4.7").preflight()
    assert caught.value.kind == "setup"
    assert "chat-completions" in str(caught.value)


@pytest.mark.asyncio
async def test_preflight_bad_effort_names_the_valid_ones():
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(_catalog_handler(), reasoning_effort="xhigh").preflight()
    assert caught.value.kind == "setup"
    assert "low" in str(caught.value) and "max" in str(caught.value)
    await client_with(_catalog_handler(), reasoning_effort="max").preflight()


@pytest.mark.asyncio
async def test_catalog_missing_fields_is_clean_error():
    broken = {"data": [{"id": "deepseek-v4.1-flash"}, "junk", {"no_id": True},
                       {"id": "deepseek-v4-pro", "api": None, "capabilities": []}]}
    with pytest.raises(OpenCodeGoError) as caught:
        await client_with(_catalog_handler(broken)).preflight()
    assert caught.value.kind == "setup"
    for unreadable in ({"data": "nope"}, ["not", "a", "dict"]):
        with pytest.raises(OpenCodeGoError) as caught:
            await client_with(_catalog_handler(unreadable)).preflight()
        assert caught.value.kind in ("transient", "setup")


def test_supported_models_filters_by_protocol():
    models = {"b": {"protocol": "chat-completions"}, "a": {"protocol": "chat-completions"},
              "c": {"protocol": "messages"}}
    parsed = {k: {"protocol": v["protocol"], "efforts": [], "output_limit": None}
              for k, v in models.items()}
    assert supported_models(parsed) == ["a", "b"]


@pytest.mark.asyncio
async def test_catalog_is_cached_and_failures_are_remembered_briefly():
    seen = []
    client = client_with(_catalog_handler(seen=seen))
    await client.list_models()
    await client.list_models()
    assert len(seen) == 1

    failing = []

    def handler(request):
        failing.append(request)
        return httpx.Response(503)

    bad = client_with(handler)
    for _ in range(2):
        with pytest.raises(OpenCodeGoError):
            await bad.list_models()
    assert len(failing) == 1


@pytest.mark.asyncio
async def test_run_clamps_max_tokens_to_the_models_output_limit():
    bodies = []
    small = catalog({"deepseek-v4.1-flash": {"output_limit": 8000}})

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=small)
        bodies.append(json.loads(request.content))
        return ok()

    client = client_with(handler)
    await client.list_models()
    await client.run("hi", timeout=30)
    assert bodies[0]["max_tokens"] == 8000


@pytest.mark.asyncio
async def test_quota_client_has_no_balance_to_read():
    quota = OpenCodeGoQuotaClient(client_with(_catalog_handler()))
    quota.invalidate()
    assert await quota.get_utilization() is None
    assert await OpenCodeGoQuotaClient().get_utilization() is None


@pytest.mark.asyncio
async def test_env_file_wins_over_a_stale_environment_value(monkeypatch):
    """`sellbird run` loads .env into the environment at startup; a key fixed in
    the file afterwards must still be the one used."""
    import sellbird.opencode_go as og
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "old-key-not-real")
    monkeypatch.setattr(og, "dotenv_values",
                        lambda *_a, **_k: {"OPENCODE_GO_API_KEY": " new-key-not-real "})
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        return ok()

    client = og.OpenCodeGoClient("deepseek-v4.1-flash", transport=httpx.MockTransport(handler))
    await client.run("hi", timeout=30)
    assert seen["auth"] == "Bearer new-key-not-real"


@pytest.mark.asyncio
async def test_blank_env_file_line_falls_back_to_the_environment(monkeypatch):
    import sellbird.opencode_go as og
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "env-key-not-real")
    monkeypatch.setattr(og, "dotenv_values", lambda *_a, **_k: {"OPENCODE_GO_API_KEY": ""})
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        return ok()

    client = og.OpenCodeGoClient("deepseek-v4.1-flash", transport=httpx.MockTransport(handler))
    await client.run("hi", timeout=30)
    assert seen["auth"] == "Bearer env-key-not-real"
