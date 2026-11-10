"""OpenCode Go across the CLI, the setup wizard and the dashboard."""

import asyncio
import json

import pytest

import sellbird.brain as brain_mod
import sellbird.cli as cli
import sellbird.config as config_mod
import sellbird.dashboard as dash
import sellbird.trainer as trainer_mod
from sellbird.config import BrainConfig
from sellbird.state import StateManager
from tests.test_brain_provider import _render_js, _render_quota_card, _status_cli
from tests.test_dashboard_signals import client  # noqa: F401  (fixture)


# ── sellbird train ──


def _train(monkeypatch, loaded, **overrides):
    seen = {}

    class FakeTrainer:
        def __init__(self, brain_config=None):
            seen["config"] = brain_config

        async def train(self, url, max_pages=100):
            pass

    monkeypatch.setattr(trainer_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(config_mod, "load_brain_config", lambda: loaded)
    args = {"url": "example.com", "max_pages": 5, "provider": None, "model": None,
            "reasoning_effort": None, **overrides}
    cli.cmd_train(type("A", (), args)())
    return seen["config"]


def test_train_accepts_opencode_go_with_the_default_model(monkeypatch):
    config = _train(monkeypatch, BrainConfig(provider="claude", model="claude-opus-4-8"),
                    provider="opencode_go")
    assert (config.provider, config.model) == ("opencode_go", "deepseek-v4.1-flash")


def test_train_model_override_for_opencode_go(monkeypatch):
    config = _train(monkeypatch, BrainConfig(provider="opencode_go"),
                    model="deepseek-v4-pro", reasoning_effort="max")
    assert (config.model, config.reasoning_effort) == ("deepseek-v4-pro", "max")


def test_train_parser_lists_opencode_go(monkeypatch):
    """`sellbird train --provider opencode_go` parses (argparse enforces choices)."""
    seen = {}
    monkeypatch.setattr(cli, "cmd_train", lambda args: seen.update(provider=args.provider,
                                                                 model=args.model))
    monkeypatch.setattr("sys.argv", ["sellbird", "train", "example.com",
                                     "--provider", "opencode_go", "--model", "deepseek-v4-pro"])
    cli.main()
    assert seen == {"provider": "opencode_go", "model": "deepseek-v4-pro"}


def test_brain_section_roundtrip():
    section = trainer_mod.brain_section(BrainConfig(provider="opencode_go"))
    assert section == {"provider": "opencode_go", "model": "deepseek-v4.1-flash"}
    assert BrainConfig(**section) == BrainConfig(provider="opencode_go")


# ── sellbird status ──


def test_status_check_without_a_key_says_what_to_set(monkeypatch, tmp_path, capsys):
    import sellbird.opencode_go as og
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setattr(og, "dotenv_values", lambda *_a, **_k: {})
    with pytest.raises(SystemExit) as exit_info:
        _status_cli(monkeypatch, tmp_path, capsys, check=True,
                    config=BrainConfig(provider="opencode_go"))
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "OpenCode Go (deepseek-v4.1-flash) NOT READY" in out
    assert "OPENCODE_GO_API_KEY" in out


def test_status_check_ready_message_and_call_counts(monkeypatch, tmp_path, capsys):
    async def ready(self):
        return ""

    monkeypatch.setattr(brain_mod.Brain, "preflight", ready)
    out = _status_cli(monkeypatch, tmp_path, capsys, check=True,
                      config=BrainConfig(provider="opencode_go"))
    assert "OpenCode Go (deepseek-v4.1-flash) ready" in out
    assert "API key" in out and "no prompt sent" in out
    assert "OpenCode Go calls today:" in out


def test_status_shows_an_opencode_go_pause(monkeypatch, tmp_path, capsys):
    out = _status_cli(
        monkeypatch, tmp_path, capsys, check=False,
        config=BrainConfig(provider="opencode_go"),
        setting=json.dumps({"provider": "opencode_go", "status": "paused",
                            "reason": "OpenCode Go rejected the API key.",
                            "since": "2026-11-10T12:00:00Z"}))
    assert "OpenCode Go PAUSED since 2026-11-10T12:00:00Z" in out


def test_usage_command_says_there_is_no_gauge(monkeypatch, tmp_path, capsys):
    import sellbird.state as state_mod
    monkeypatch.setattr(config_mod, "load_brain_config",
                        lambda: BrainConfig(provider="opencode_go"))
    monkeypatch.setattr(state_mod, "StateManager", lambda: StateManager(str(tmp_path / "u.db")))
    cli.cmd_usage(type("A", (), {"days": 30})())
    out = capsys.readouterr().out
    assert "OpenCode Go Usage" in out
    assert "no documented balance endpoint" in out


# ── dashboard ──


def test_dashboard_quota_client_for_opencode_go_reads_nothing():
    client_ = dash._quota_client_for("opencode_go", "deepseek-v4.1-flash")
    assert type(client_).__name__ == "OpenCodeGoQuotaClient"
    assert asyncio.run(client_.get_utilization()) is None


def test_dashboard_usage_marks_opencode_go_quota_unavailable(client, monkeypatch):
    monkeypatch.setattr(dash, "_active_brain", lambda: BrainConfig(provider="opencode_go"))
    data = client.get("/api/usage").json()
    assert (data["quota_provider"], data["quota_status"]) == ("opencode_go", "unavailable")


def test_dashboard_stats_report_opencode_go_calls(client, monkeypatch):
    asyncio.run(StateManager(str(client.db_path)).init_db())
    asyncio.run(StateManager(str(client.db_path)).increment_usage(provider="opencode_go"))
    monkeypatch.setattr(dash, "_active_brain", lambda: BrainConfig(provider="opencode_go"))
    stats = client.get("/api/stats").json()
    assert stats["opencode_go_calls_today"] == 1
    assert stats["brain_provider"] == "opencode_go"


def test_provider_label_names_the_paused_brain():
    assert dash._provider_label("opencode_go") == "OpenCode Go"
    assert dash._provider_label("codex") == "Codex"
    assert dash._provider_label("") == "Sellbird's model"


def test_call_chips_show_opencode_go_first_and_say_when_paused():
    data = {"brain_provider": "opencode_go", "opencode_go_calls_today": 4,
            "claude_calls_today": 0, "codex_calls_today": 0, "brain_status": {}}
    html = _render_js("brainCallChips", data)
    assert "OpenCode Go calls today <b>4</b>" in html
    assert "Claude calls today" not in html
    paused = _render_js("brainCallChips", {**data, "brain_status": {
        "status": "paused", "provider": "opencode_go", "reason": "bad key"}})
    assert "OpenCode Go <b>paused</b>" in paused


def test_quota_card_for_opencode_go_explains_the_missing_gauge():
    html = _render_quota_card({"quota_provider": "opencode_go", "quota_status": "unavailable",
                               "quota": None})
    assert "OpenCode Go Quota" in html
    assert "no documented balance endpoint" in html
    assert "progress-bar" not in html


# ── sellbird setup ──


def test_setup_offers_opencode_go_with_the_default_model(monkeypatch):
    import sellbird.setup as setup_mod
    answers = iter(["opencode_go", ""])  # provider, then Enter for the suggested model
    asked = []

    def fake_ask(prompt, default="", required=True, secret=False):
        asked.append((prompt, default))
        answer = next(answers)
        return answer or default

    monkeypatch.setattr(setup_mod, "_ask", fake_ask)
    config = setup_mod._ask_brain_config()
    assert (config.provider, config.model) == ("opencode_go", "deepseek-v4.1-flash")
    assert "opencode_go" in asked[0][0]
    assert asked[1][1] == "deepseek-v4.1-flash"  # the default is the suggestion


def test_setup_stores_the_key_hidden_and_checks_it(monkeypatch, capsys):
    import sellbird.setup as setup_mod

    asked = {}

    def fake_ask(prompt, default="", required=True, secret=False):
        asked.update(prompt=prompt, secret=secret)
        return "  key-not-real  "

    class FakeBrain:
        def __init__(self, state, config):
            asked["config"] = config

        async def preflight(self):
            return ""

    class FakeState:
        async def init_db(self):
            pass

    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)  # restored after the test
    monkeypatch.setattr(setup_mod, "_ask", fake_ask)
    monkeypatch.setattr(setup_mod, "Brain", FakeBrain)
    monkeypatch.setattr(setup_mod, "StateManager", FakeState)
    monkeypatch.setattr(setup_mod, "_read_env_file", lambda: {})
    monkeypatch.setattr(setup_mod.SetupWizard, "_check_python_packages", lambda self: None)

    wizard = setup_mod.SetupWizard()
    wizard.brain_config = BrainConfig(provider="opencode_go")
    asyncio.run(wizard._check_opencode_go())

    assert asked["secret"] is True  # typed without echo
    assert wizard.env_vars["OPENCODE_GO_API_KEY"] == "key-not-real"
    out = capsys.readouterr().out
    assert "key-not-real" not in out
    assert "designed for OpenCode" in out or "coding agents" in out  # the intended-use notice
    assert "Ready" in out


def test_setup_reuses_a_saved_key_and_stops_on_a_bad_one(monkeypatch, capsys):
    import sellbird.setup as setup_mod

    class FakeBrain:
        def __init__(self, state, config):
            pass

        async def preflight(self):
            return "OpenCode Go rejected the API key."

    class FakeState:
        async def init_db(self):
            pass

    def never_ask(*a, **k):
        if k.get("secret"):
            raise AssertionError("must not ask for a key that is already saved")
        return False  # "Continue anyway?" -> no

    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setattr(setup_mod, "_ask", never_ask)
    monkeypatch.setattr(setup_mod, "_ask_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(setup_mod, "Brain", FakeBrain)
    monkeypatch.setattr(setup_mod, "StateManager", FakeState)
    monkeypatch.setattr(setup_mod, "_read_env_file",
                        lambda: {"OPENCODE_GO_API_KEY": "saved-key-not-real"})
    wizard = setup_mod.SetupWizard()
    wizard.brain_config = BrainConfig(provider="opencode_go")
    with pytest.raises(SystemExit):
        asyncio.run(wizard._check_opencode_go())
    assert "rejected the API key" in capsys.readouterr().out
