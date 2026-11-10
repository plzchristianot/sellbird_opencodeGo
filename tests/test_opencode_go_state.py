"""Per-provider usage for OpenCode Go: counters, usage events, and the schema upgrade."""

import os
import sqlite3
import tempfile
from datetime import date

import pytest
import pytest_asyncio

from sellbird.state import MIGRATIONS, StateManager


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


@pytest.mark.asyncio
async def test_increment_and_read_opencode_go_usage(state):
    await state.increment_usage(provider="opencode_go")
    assert await state.get_usage_today("opencode_go") == 1
    assert await state.get_usage_today("claude") == 0
    assert await state.get_usage_today("codex") == 0


@pytest.mark.asyncio
async def test_usage_event_accepts_provider(state):
    stored = await state.record_usage_event(
        agent="writer", task="draft", model="deepseek-v4.1-flash",
        input_tokens=39, output_tokens=19, cost_usd=0.00001035,
        source="opencode_go_http", provider="opencode_go",
    )
    assert stored is True
    rows = await state.usage_by_provider(days=1)
    assert [r["provider"] for r in rows] == ["opencode_go"]
    assert rows[0]["calls"] == 1


@pytest.mark.asyncio
async def test_summary_reports_opencode_go_today(state):
    await state.increment_usage(provider="opencode_go")
    await state.increment_usage(provider="opencode_go")
    summary = await state.get_state_summary()
    assert summary["usage_today_by_provider"] == {"claude": 0, "codex": 0, "opencode_go": 2}


@pytest.mark.asyncio
async def test_v16_database_upgrades_with_counts_intact():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "old.db")
        db = sqlite3.connect(path)
        for script in MIGRATIONS[:16]:
            db.executescript(script)
        db.execute("PRAGMA user_version = 16")
        db.execute(
            "INSERT INTO usage_log (id, date, claude_calls, codex_calls) VALUES ('u1', ?, 3, 2)",
            (date.today().isoformat(),),
        )
        db.commit()
        db.close()

        sm = StateManager(path)
        await sm.init_db()
        assert await sm.get_usage_today("claude") == 3
        assert await sm.get_usage_today("codex") == 2
        assert await sm.get_usage_today("opencode_go") == 0
