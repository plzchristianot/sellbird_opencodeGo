"""Live check against the real OpenCode Go service. Skipped unless you opt in:

    SELLBIRD_OPENCODE_GO_LIVE=1 OPENCODE_GO_API_KEY=... pytest tests/test_opencode_go_live.py

It makes the catalog request and two tiny completions on your plan. It prints
counts only, never the key or a reply.
"""

import os

import pytest

from sellbird.config import OPENCODE_GO_DEFAULT_MODEL
from sellbird.opencode_go import OpenCodeGoClient, supported_models

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SELLBIRD_OPENCODE_GO_LIVE") == "1"
         and os.environ.get("OPENCODE_GO_API_KEY", "").strip()),
    reason="set SELLBIRD_OPENCODE_GO_LIVE=1 and OPENCODE_GO_API_KEY to run against the live service",
)

PROMPT = 'Reply with exactly: {"ok":true}'


@pytest.mark.asyncio
async def test_live_catalog_and_default_model_preflight():
    client = OpenCodeGoClient(OPENCODE_GO_DEFAULT_MODEL)
    models = await client.list_models()
    assert OPENCODE_GO_DEFAULT_MODEL in supported_models(models)
    await client.preflight()  # no prompt sent


@pytest.mark.asyncio
async def test_live_short_completion_returns_text_and_cost():
    result = await OpenCodeGoClient(OPENCODE_GO_DEFAULT_MODEL).run(PROMPT, timeout=90)
    assert "ok" in result.text
    assert result.usage["output_tokens"] > 0
    assert result.usage["cost_usd"] >= 0


@pytest.mark.asyncio
async def test_live_reasoning_effort_is_accepted():
    """Confirms the catalog's effort names work as the request's reasoning_effort."""
    client = OpenCodeGoClient(OPENCODE_GO_DEFAULT_MODEL, reasoning_effort="low")
    await client.preflight()
    result = await client.run(PROMPT, timeout=90)
    assert "ok" in result.text
