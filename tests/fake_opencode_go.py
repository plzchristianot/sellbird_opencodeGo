"""Builders for OpenCode Go tests: fictional replies and a client on a mock transport.

Nothing here talks to the network, and the key is a placeholder.
"""

import httpx

TEST_KEY = "test-key-not-real"


def completion(content="ok", reasoning="", finish="stop", tool_calls=None,
               allowance=0.00001, model="deepseek-v4.1-flash") -> dict:
    """A chat-completions reply shaped like the live service's."""
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "gw-test", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {
            "prompt_tokens": 39, "completion_tokens": 19, "total_tokens": 58,
            "prompt_tokens_details": {"cached_tokens": 0},
            "opencode": {"billing_source": "go", "allowance": allowance,
                         "tokens": {"input": 39, "output": 19, "reasoning": 12,
                                    "cache_read": 0, "cache_write_5m": 0,
                                    "cache_write_1h": 0}},
        },
    }


def ok(payload: dict | None = None) -> httpx.Response:
    return httpx.Response(200, json=payload if payload is not None else completion())


def client_with(handler, **kw):
    """An OpenCodeGoClient whose HTTP goes to ``handler(request) -> Response``."""
    from sellbird.opencode_go import OpenCodeGoClient

    kw.setdefault("api_key", TEST_KEY)
    return OpenCodeGoClient(kw.pop("model", "deepseek-v4.1-flash"),
                            kw.pop("reasoning_effort", ""),
                            transport=httpx.MockTransport(handler), **kw)


def catalog(models: dict[str, dict]) -> dict:
    """A /models reply. ``models`` maps id -> {"protocol", "efforts", "output_limit"}."""
    data = []
    for model_id, spec in models.items():
        entry = {"id": model_id, "object": "model", "type": "chat",
                 "api": {spec.get("protocol", "chat-completions"):
                         {"base_url": "https://opencode.ai/inference/go/openai/v1"}},
                 "capabilities": {"reasoning": {"supported": True,
                                                "effort": spec.get("efforts", [])}},
                 "limit": {"context": 1000000, "output": spec.get("output_limit", 384000)}}
        data.append(entry)
    return {"object": "list", "data": data}


DEFAULT_CATALOG = {
    "deepseek-v4.1-flash": {"efforts": ["low", "high", "max"]},
    "deepseek-v4-pro": {"efforts": ["high", "max"]},
    "grok-4.7": {"protocol": "responses", "efforts": ["low", "high"]},
}
