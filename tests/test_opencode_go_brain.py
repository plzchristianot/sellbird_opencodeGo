"""Brain on the OpenCode Go provider: no other CLI is touched, failures pause
the right way, and usage is counted on its own."""

import asyncio
import json
import os
import tempfile

import pytest
import pytest_asyncio

import sellbird.brain as brain_mod
from sellbird.brain import Brain
from sellbird.config import BrainConfig
from sellbird.opencode_go import OpenCodeGoError, OpenCodeGoResult
from sellbird.state import StateManager


@pytest.fixture(autouse=True)
def _no_cli_ever(monkeypatch):
    """An OpenCode Go brain must never start the Claude or Codex CLI. If the
    wiring is wrong, fail fast instead of launching a real one."""
    async def refuse(*a, **k):
        raise AssertionError("an OpenCode Go brain must not start any CLI")

    monkeypatch.setattr(brain_mod.asyncio, "create_subprocess_exec", refuse)
    monkeypatch.setattr(brain_mod, "RETRY_BASE_DELAY", 0)


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


class _Stub:
    """Stands in for OpenCodeGoClient, like _StubCodex does for Codex."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.preflight_outcomes = []
        self.preflights = 0
        self.print = ("opencode_go", "key-v1", "deepseek-v4.1-flash")

    def fingerprint(self):
        return self.print

    async def preflight(self):
        self.preflights += 1
        if self.preflight_outcomes:
            outcome = self.preflight_outcomes.pop(0)
            if outcome is not None:
                raise outcome

    async def run(self, prompt, timeout, skip_if=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _result(text="hello", cost=0.00001):
    return OpenCodeGoResult(
        text=text, model="deepseek-v4.1-flash", thread_id="sellbird-abc", duration_ms=900,
        usage={"input_tokens": 39, "output_tokens": 19, "cache_read_tokens": 0,
               "cache_creation_tokens": 0, "cost_usd": cost})


def _brain(state, outcomes):
    brain = Brain(state, BrainConfig(provider="opencode_go"))
    brain.opencode_go = _Stub(outcomes)
    return brain


def test_brain_builds_the_opencode_go_client_with_the_default_model(state):
    brain = Brain(state, BrainConfig(provider="opencode_go", reasoning_effort="low"))
    assert brain.provider_label == "OpenCode Go"
    assert brain.opencode_go.model == "deepseek-v4.1-flash"
    assert brain.opencode_go.reasoning_effort == "low"
    assert brain.codex is None
    assert type(brain.quota).__name__ == "OpenCodeGoQuotaClient"


@pytest.mark.asyncio
async def test_never_touches_claude_or_codex(state, monkeypatch):
    async def no_process(*a, **k):
        raise AssertionError("an OpenCode Go brain must not start any CLI")

    monkeypatch.setattr(brain_mod.asyncio, "create_subprocess_exec", no_process)
    brain = _brain(state, [_result("hello", cost=0.00001035)])
    assert await brain.think("hi", agent="writer", task="draft") == "hello"

    assert await state.get_usage_today("opencode_go") == 1
    assert await state.get_usage_today("claude") == 0
    assert await state.get_usage_today("codex") == 0
    async with state._connect() as db:
        async with db.execute(
            "SELECT provider, source, agent, task, model, cost_usd FROM usage_events"
        ) as cursor:
            rows = await cursor.fetchall()
    assert rows == [("opencode_go", "opencode_go_http", "writer", "draft",
                     "deepseek-v4.1-flash", 0.00001035)]


@pytest.mark.asyncio
async def test_think_json_parses_fenced_json(state):
    brain = _brain(state, [_result('```json\n{"intent": "interested"}\n```')])
    assert await brain.think_json("classify") == {"intent": "interested"}


@pytest.mark.asyncio
async def test_persistent_failure_pauses_and_skips_further_calls(state):
    brain = _brain(state, [OpenCodeGoError("bad key", kind="auth")])
    assert await brain.think("one") == ""
    assert await brain.think("two") == ""
    assert brain.opencode_go.calls == 1
    assert brain.paused_reason == "bad key"
    stored = json.loads(await state.get_setting(brain_mod.BRAIN_STATUS_KEY))
    assert (stored["provider"], stored["status"], stored["kind"]) == ("opencode_go", "paused", "auth")


@pytest.mark.asyncio
async def test_usage_limit_pauses_and_closes_the_budget(state):
    brain = _brain(state, [OpenCodeGoError("limit", kind="usage_limit")])
    assert await brain.think("hi") == ""
    stored = json.loads(await state.get_setting(brain_mod.BRAIN_STATUS_KEY))
    assert stored["kind"] == "usage_limit"
    assert await brain.is_within_budget(1000, max_percent=80) is False
    assert "OpenCode Go usage limit" in brain.paused_reason


@pytest.mark.asyncio
async def test_retries_are_bounded(state, monkeypatch):
    monkeypatch.setattr(brain_mod, "RETRY_BASE_DELAY", 0)
    err = OpenCodeGoError("busy", kind="transient", retryable=True)
    brain = _brain(state, [err, err, err, _result("late")])
    assert await brain.think("hi", max_retries=2) == ""
    assert brain.opencode_go.calls == 3

    brain = _brain(state, [err, _result("recovered")])
    assert await brain.think("hi", max_retries=2) == "recovered"


@pytest.mark.asyncio
async def test_unexpected_errors_return_empty_without_retry(state):
    brain = _brain(state, [RuntimeError("detail that must not be logged")])
    assert await brain.think("hi") == ""
    assert brain.opencode_go.calls == 1


@pytest.mark.asyncio
async def test_cancellation_propagates(state):
    brain = _brain(state, [asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await brain.think("hi")


@pytest.mark.asyncio
async def test_budget_uses_the_call_counter_when_there_is_no_gauge(state):
    brain = _brain(state, [_result()])
    assert await brain.is_within_budget(1, max_percent=80) is True
    await state.increment_usage(provider="claude")  # other providers don't count
    assert await brain.is_within_budget(1, max_percent=80) is True
    await brain.think("hi")
    assert await brain.is_within_budget(1, max_percent=80) is False


@pytest.mark.asyncio
async def test_preflight_pause_is_sticky_until_the_key_changes(state):
    brain = _brain(state, [])
    brain.opencode_go.preflight_outcomes = [OpenCodeGoError("bad key", kind="auth"), None]

    assert await brain.preflight() == "bad key"
    assert await brain.preflight() == "bad key"  # nothing changed: no re-check
    assert brain.opencode_go.preflights == 1

    brain.opencode_go.print = ("opencode_go", "key-v2", "deepseek-v4.1-flash")
    assert await brain.preflight() == ""
    assert brain.opencode_go.preflights == 2
    stored = json.loads(await state.get_setting(brain_mod.BRAIN_STATUS_KEY))
    assert stored["status"] == "ok"


@pytest.mark.asyncio
async def test_transient_preflight_failure_is_not_sticky(state):
    brain = _brain(state, [])
    brain.opencode_go.preflight_outcomes = [
        OpenCodeGoError("OpenCode Go call timed out.", kind="transient", retryable=True), None]
    assert await brain.preflight() == "OpenCode Go call timed out."
    assert await brain.preflight() == ""
    assert brain.opencode_go.preflights == 2
    assert await state.get_setting(brain_mod.BRAIN_STATUS_KEY) == ""


@pytest.mark.asyncio
async def test_a_pause_stored_for_another_provider_is_ignored(state):
    await state.set_setting(brain_mod.BRAIN_STATUS_KEY, json.dumps(
        {"provider": "claude", "status": "paused", "reason": "CLI too old", "kind": "setup"}))
    raw = await state.get_setting(brain_mod.BRAIN_STATUS_KEY)
    assert brain_mod.active_brain_status(raw, "opencode_go") == {}

    brain = _brain(state, [_result("fine")])
    assert await brain.preflight() == ""
    assert brain.paused_reason == ""
    assert await brain.think("hi") == "fine"


@pytest.mark.asyncio
async def test_another_providers_usage_limit_pause_is_not_inherited(state):
    """Switching from Codex: its stored 'usage limit until later' says nothing
    about the OpenCode Go plan, so this Brain must not restore it."""
    import time
    await state.set_setting(brain_mod.BRAIN_STATUS_KEY, json.dumps(
        {"provider": "codex", "status": "paused", "kind": "usage_limit",
         "reason": "Codex usage limit reached", "until": time.time() + 3000}))
    brain = _brain(state, [_result("fine")])
    assert await brain.preflight() == ""
    assert brain.paused_reason == ""
    assert await brain.is_within_budget(1000, max_percent=80) is True
    assert await brain.think("hi") == "fine"
