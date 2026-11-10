# OpenCode Go Brain Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `brain.provider: opencode_go` run all of Sellbird's model calls on an OpenCode Go subscription, text-only, defaulting to `deepseek-v4.1-flash`.

**Architecture:** A new adapter `sellbird/opencode_go.py` mirrors `CodexClient`'s contract (`preflight()`, `run()`, `fingerprint()`, a quota client) and raises an error type with the same `kind` values as `CodexError`. `Brain` gains a shared `_think_adapter` used by Codex and OpenCode Go, so pausing, retries and usage accounting stay in one place. Claude's path is untouched.

**Tech Stack:** Python >= 3.11, `httpx` (already a dependency; tests use `httpx.MockTransport`), pydantic, aiosqlite, pytest + pytest-asyncio. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-10-opencode-go-brain-design.md`

## Global Constraints

- Provider id is exactly `opencode_go`; label `"OpenCode Go"`; env var `OPENCODE_GO_API_KEY`.
- Default model `deepseek-v4.1-flash` (constant `OPENCODE_GO_DEFAULT_MODEL`); `brain.model` / `--model` override it.
- Base URL fixed in code: `https://opencode.ai/inference/go/openai/v1`; endpoints `/models` and `/chat/completions`; redirects disabled; no other host.
- Request body: one `user` message, `model`, `max_tokens`, optional `reasoning_effort`. **Never** `tools`, `tool_choice` or `stream`.
- Headers: `Authorization: Bearer <key>`, `User-Agent: sellbird/<version>` (from `importlib.metadata`, fallback `0`), `x-opencode-session: <stable id>`. Never imitate OpenCode or a coding agent.
- v1 supports only models whose catalog `api` has a `chat-completions` entry.
- Never fall back to Claude, Codex or any other provider; existing Claude and Codex behaviour and tests must stay green.
- The key never appears in logs, exceptions, test output or fixtures. Fixtures use fictional data only (CONTRIBUTING.md).
- Commit trailers on every commit: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_017PCPjxismpEDLxfJaiW3Zs`. Work on branch `claude/compassionate-turing-zb6ffs`.

## Review Focus

Failure modes the spec implies that no happy path exercises, most likely first. Each has a test in the owning task.

1. The server (or a proxy) echoes the `Authorization` header or key in an error body: it must not reach logs or exception text. (Task 3)
2. A reply with empty `content` but non-empty `reasoning_content` must not be returned as the answer. (Task 3)
3. `.env` key with surrounding whitespace/newline or only whitespace: stripped; blank counts as missing. (Task 3)
4. A 200 response that is not JSON (captive portal HTML), or a catalog entry missing `api`/`capabilities`: a clean `transient`/`setup` error, never a `KeyError`. (Tasks 3, 4)
5. A `brain_status` pause stored for another provider is ignored after switching to `opencode_go`, and an existing database at schema version 16 upgrades with Claude/Codex counts intact. (Tasks 2, 5)

---

### Task 1: Config — provider, default model, effort values

**Files:**
- Modify: `sellbird/config.py:633-681` (`BRAIN_PROVIDERS`, `BrainConfig`)
- Test: `tests/test_opencode_go_config.py` (create)

**Interfaces:**
- Produces: `BRAIN_PROVIDERS = ("claude", "codex", "opencode_go")`; `OPENCODE_GO_DEFAULT_MODEL: str`; `OPENCODE_GO_REASONING_EFFORTS: tuple[str, ...]` = `("none", "minimal", "low", "medium", "high", "xhigh", "max")`; `BrainConfig(provider="opencode_go")` has `model == OPENCODE_GO_DEFAULT_MODEL` when none is given.

- [ ] **Step 1: Set up the environment.** Run `python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`. Run `pytest tests/test_brain_provider.py tests/test_config.py -q`; record that it passes as the baseline.
- [ ] **Step 2: Write failing tests** in `tests/test_opencode_go_config.py`: `test_opencode_go_is_a_provider` (`BrainConfig(provider="opencode_go")` builds); `test_default_model_is_deepseek` (`.model == "deepseek-v4.1-flash"`); `test_explicit_model_wins` (`model="deepseek-v4-pro"` kept); `test_max_effort_allowed_only_for_opencode_go` (`reasoning_effort="max"` valid for `opencode_go`, `ValidationError` for `codex`); `test_claude_and_codex_models_unchanged` (empty model stays `""` for both).
- [ ] **Step 3: Run** `pytest tests/test_opencode_go_config.py -q`; expect FAIL (unknown provider).
- [ ] **Step 4: Implement** the constants and, in `BrainConfig`, a `model_validator(mode="after")` that fills the default model for `opencode_go` and validates `reasoning_effort` against `OPENCODE_GO_REASONING_EFFORTS` for that provider (the existing field validator must stop rejecting `max`/`none` before provider is known: move the effort-set check into the model validator for all providers, keeping Codex/Claude's current allowed set and error text).
- [ ] **Step 5: Run** `pytest tests/test_opencode_go_config.py tests/test_brain_provider.py tests/test_config.py -q`; expect all PASS.
- [ ] **Step 6: Commit** `feat(config): add opencode_go brain provider and default model`.

### Task 2: State — migration and per-provider usage

**Files:**
- Modify: `sellbird/state.py` (append migration v17 after v16 at ~line 549; `BRAIN_PROVIDERS` and `_USAGE_LOG_COLUMN` at ~557; the `usage_log` read and `usage_today_by_provider` at ~2765-2787; any other place that enumerates `"codex"`, found with `grep -n 'codex' sellbird/state.py`)
- Test: `tests/test_opencode_go_state.py` (create)

**Interfaces:**
- Produces: `StateManager.increment_usage(provider="opencode_go")`, `get_usage_today("opencode_go")`, `record_usage_event(..., provider="opencode_go")`, and `summary["usage_today_by_provider"]["opencode_go"]`.

- [ ] **Step 1: Write failing tests:** `test_increment_and_read_opencode_go_usage` (count 1 for `opencode_go`, 0 for claude and codex); `test_usage_event_accepts_provider` (a row with `provider="opencode_go"`, `cost_usd=0.00001` shows in `usage_by_provider(days=1)`); `test_v16_database_upgrades` (create a DB, set `PRAGMA user_version = 16` with a claude and a codex count already stored in `usage_log`, reopen with `init_db()`, assert both counts intact and `opencode_go_calls` readable as 0). Use the `state` fixture pattern from `tests/test_brain_provider.py`.
- [ ] **Step 2: Run** `pytest tests/test_opencode_go_state.py -q`; expect FAIL (`KeyError`).
- [ ] **Step 3: Implement** migration v17 `ALTER TABLE usage_log ADD COLUMN opencode_go_calls INTEGER DEFAULT 0;` (never edit earlier migrations), extend the two provider tables, and report the third count in `usage_today_by_provider`.
- [ ] **Step 4: Run** `pytest tests/test_opencode_go_state.py tests/test_state.py tests/test_state_extra.py -q`; expect PASS.
- [ ] **Step 5: Commit** `feat(state): count OpenCode Go calls per provider`.

### Task 3: Adapter core — `OpenCodeGoClient.run`

**Files:**
- Create: `sellbird/opencode_go.py`, `tests/fake_opencode_go.py`
- Test: `tests/test_opencode_go.py` (create)

**Interfaces:**
- Produces in `sellbird/opencode_go.py`:
  - constants `BASE_URL = "https://opencode.ai/inference/go/openai/v1"`, `MODELS_URL = "https://opencode.ai/inference/go/v1/models"` (both verified live 2026-11-10), `API_KEY_ENV = "OPENCODE_GO_API_KEY"`, `MAX_TOKENS = 16000`, `RETRY_MAX_TOKENS = 32000`.
  - `class OpenCodeGoError(Exception)`: `__init__(self, message: str, kind: str = "transient", retryable: bool = False)`; kinds `auth | setup | usage_limit | transient | paused`.
  - `@dataclass class OpenCodeGoResult`: `text: str`, `model: str = ""`, `thread_id: str = ""` (the `x-opencode-session` used), `duration_ms: int = 0`, `usage: dict` (keys `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `cost_usd` where `cost_usd` is `opencode.allowance`), `rate_limits: None`.
  - `class OpenCodeGoClient`: `__init__(self, model: str, reasoning_effort: str = "", *, api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None)`; `fingerprint(self) -> tuple`; `async run(self, prompt: str, timeout: float, skip_if=None) -> OpenCodeGoResult`.
  - `api_key` resolution order: constructor arg, `os.environ[API_KEY_ENV]`, `dotenv_values(PROJECT_ROOT / ".env")`; always `.strip()`; blank means missing → `OpenCodeGoError(kind="setup")` naming `OPENCODE_GO_API_KEY` (never its value).
- `tests/fake_opencode_go.py` produces: `completion(content="ok", reasoning="", finish="stop", tool_calls=None, allowance=0.00001) -> dict` and `client_with(handler, **kw) -> OpenCodeGoClient` (wraps `httpx.MockTransport(handler)`, key `"test-key-not-real"`).
- Error mapping (the tests pin it): 401/403 → `auth`; 400 whose body mentions an unavailable/unknown model, or any missing-key condition → `setup`; 429 or a body mentioning `limit`/`quota`/`allowance` → `usage_limit`; 5xx, `httpx.TimeoutException`, `httpx.TransportError`, non-JSON 200 → `transient` with `retryable=True`.

- [ ] **Step 1: Write failing tests** in `tests/test_opencode_go.py` (all `@pytest.mark.asyncio`, handlers assert on `request`):
  - `test_run_returns_content_only_and_usage`: reply has `content="{\"ok\":true}"`, `reasoning_content="thinking"`; result `.text == '{"ok":true}'` and `.usage["cost_usd"] == 0.00001`.
  - `test_request_shape`: URL is `BASE_URL + "/chat/completions"`; headers `authorization == "Bearer test-key-not-real"`, `user-agent` starts `sellbird/`, `x-opencode-session` non-empty and identical across two calls; JSON body has exactly one `user` message and none of `tools`, `tool_choice`, `stream`.
  - `test_error_mapping` (parametrised): 401→auth, 403→auth, 429→usage_limit, 400 `"Model is unavailable"`→setup, 400 mentioning a usage limit→usage_limit, 500→transient+retryable, `httpx.ReadTimeout`→transient+retryable.
  - `test_tool_calls_rejected`: reply with `tool_calls=[...]` raises `OpenCodeGoError` (not retryable).
  - `test_length_finish_retries_once_then_fails`: first reply `finish="length"` triggers a second request with `max_tokens == RETRY_MAX_TOKENS`; second `length` raises `transient`.
  - `test_redirect_not_followed`: a 302 raises `OpenCodeGoError`; the handler is never called for the redirect target.
  - `test_reasoning_only_reply_is_not_an_answer` (Review Focus 2): empty `content` with reasoning raises `transient`.
  - `test_key_never_leaks` (Review Focus 1): server returns 500 with the bearer value in the body; the exception text and `caplog` contain neither the key nor `Bearer`.
  - `test_blank_or_padded_key` (Review Focus 3): `" test-key-not-real\n"` works and sends the stripped value; `"   "` raises `setup`.
  - `test_non_json_200_is_transient` (Review Focus 4).
  - `test_skip_if_pauses`: `skip_if=lambda: "paused"` raises `kind="paused"` without a request.
  - `test_cancellation_propagates`: a handler that raises `asyncio.CancelledError` is not swallowed.
- [ ] **Step 2: Run** `pytest tests/test_opencode_go.py -q`; expect FAIL (module missing).
- [ ] **Step 3: Implement** `opencode_go.py` and `fake_opencode_go.py` as specified above. One `httpx.AsyncClient(follow_redirects=False, trust_env=...)` per call; `trust_env` stays default so the environment's proxy settings apply. Session id: `"sellbird-" + uuid4().hex[:16]` created once per client instance. Errors never include response bodies, headers or the prompt; use `logsafety.redact` on any logged text.
- [ ] **Step 4: Run** `pytest tests/test_opencode_go.py -q`; expect all PASS.
- [ ] **Step 5: Commit** `feat: add OpenCode Go chat-completions adapter`.

### Task 4: Adapter — catalog, preflight, quota client

**Files:**
- Modify: `sellbird/opencode_go.py`
- Test: `tests/test_opencode_go.py`; extend `tests/fake_opencode_go.py` with `catalog(models: dict[str, dict]) -> dict` where each value is `{"protocol": "chat-completions", "efforts": ["low","high"], "output_limit": 384000}`.

**Interfaces:**
- Consumes: Task 3's `OpenCodeGoClient`, `OpenCodeGoError`.
- Produces: `async OpenCodeGoClient.list_models(self) -> dict[str, dict]` (id → `{"protocol": str, "efforts": list[str]}`; cached 300 s, failures 60 s); `async OpenCodeGoClient.preflight(self) -> None`; `supported_models(catalog) -> list[str]` (module function, ids whose protocol is `chat-completions`, sorted); `class OpenCodeGoQuotaClient` with `__init__(self, client=None)`, `invalidate(self) -> None`, `async get_utilization(self) -> None`.

- [ ] **Step 1: Write failing tests:** `test_preflight_success_sends_no_prompt` (only `GET /models` requests seen); `test_preflight_missing_key` → `setup`; `test_preflight_bad_key` (401) → `auth`; `test_preflight_unknown_model_lists_supported_ids` (message contains a supported id, kind `setup`); `test_preflight_wrong_protocol` (`responses` model → `setup`); `test_preflight_bad_effort` (effort not in the model's list → `setup`, message names the valid ones); `test_catalog_missing_fields_is_clean_error` (entry without `api`/`capabilities` → `setup`, no `KeyError`; Review Focus 4); `test_supported_models_filters_by_protocol`; `test_quota_client_returns_none`.
- [ ] **Step 2: Run** the new tests; expect FAIL.
- [ ] **Step 3: Implement** catalog parsing defensively with `.get()`; fetch `MODELS_URL` with the same headers and error mapping as `run()`. A model's protocol is the first key of its `api` object; its efforts are `capabilities.reasoning.effort`. In `run()`, clamp `max_tokens` to a cached model's `output_limit` when the catalog is already loaded.
- [ ] **Step 4: Run** `pytest tests/test_opencode_go.py -q`; expect PASS.
- [ ] **Step 5: Commit** `feat: OpenCode Go catalog preflight and quota stub`.

### Task 5: Brain integration

**Files:**
- Modify: `sellbird/brain.py` (`PROVIDER_LABELS`, `__init__`, `preflight`, `_fingerprint`, `_think_codex` → shared helper, `_record_codex_usage`)
- Test: `tests/test_brain_provider.py` (extend), reuse `tests/fake_opencode_go.py`

**Interfaces:**
- Consumes: `OpenCodeGoClient`, `OpenCodeGoError`, `OpenCodeGoQuotaClient`, `OPENCODE_GO_DEFAULT_MODEL`.
- Produces: `Brain(state, BrainConfig(provider="opencode_go")).opencode_go` (client, replaceable by a stub exactly like `brain.codex`), `brain.quota` is an `OpenCodeGoQuotaClient`, `brain.provider_label == "OpenCode Go"`, `Brain._think_adapter(self, client, prompt, session_id, timeout, max_retries, agent, task, source) -> str`.

- [ ] **Step 1: Write failing tests** (a `_StubOpenCodeGo` mirroring `_StubCodex`): `test_opencode_go_brain_never_touches_claude_or_codex` (patch `create_subprocess_exec` to fail; `think` returns text; `get_usage_today("opencode_go") == 1`, claude and codex 0; the usage row has `provider="opencode_go"`, `source="opencode_go_http"`, `cost_usd == allowance`); `test_think_json_parses_fenced_json`; `test_persistent_failure_pauses_then_skips` (an `auth` error calls `_mark_unavailable`, a second `think` makes no client call); `test_usage_limit_pauses_an_hour_and_closes_budget`; `test_retries_are_bounded`; `test_budget_uses_call_counter` (`is_within_budget(max_daily_calls=1, max_percent=80)` is False after one call); `test_preflight_pause_sticky_until_fingerprint_changes`; `test_pause_for_other_provider_ignored` (a stored `brain_status` for `claude` does not mark `opencode_go` unavailable; Review Focus 5); `test_default_model_reaches_client` (`brain.model == "deepseek-v4.1-flash"`).
- [ ] **Step 2: Run** `pytest tests/test_brain_provider.py -q`; expect the new tests FAIL, old ones PASS.
- [ ] **Step 3: Implement.** Extract the Codex retry/pause loop into `_think_adapter` with the usage-recording differences (counter provider, recorder, log label) passed in; the Codex path calls it with identical behaviour. In `preflight`, generalise `except CodexError` to both error types (persistent kinds unchanged). `_fingerprint` for `opencode_go` returns `client.fingerprint()` (key present flag and model, never the key).
- [ ] **Step 4: Run** `pytest tests/test_brain_provider.py tests/test_codex.py tests/test_codex_e2e.py tests/test_codex_soak.py tests/test_opencode_go.py -q`; expect all PASS (Codex behaviour unchanged).
- [ ] **Step 5: Commit** `feat(brain): run Brain on OpenCode Go through a shared adapter loop`.

### Task 6: CLI, setup wizard, dashboard

**Files:**
- Modify: `sellbird/cli.py` (~237 usage line, ~296-364 status/usage, ~1142 `--provider`), `sellbird/setup.py` (~112-165, ~242-262, ~852), `sellbird/dashboard.py` (~960-985, ~2247), `sellbird/trainer.py:326` (verify only)
- Test: `tests/test_opencode_go_surfaces.py` (create); extend `tests/test_trainer_config.py`

**Interfaces:**
- Consumes: `PROVIDER_LABELS`, `OpenCodeGoQuotaClient`, `Brain.preflight()`.
- Produces: `sellbird train --provider opencode_go [--model ID]`; `sellbird status` prints `OpenCode Go calls today: N`; setup option `opencode_go` that stores `OPENCODE_GO_API_KEY` in `self.env_vars` (read with `getpass`, never echoed) and runs `Brain.preflight()`; dashboard labels from `PROVIDER_LABELS` instead of the `"Codex" if ... else` branches; `_quota_client_for("opencode_go", ...)` returns an `OpenCodeGoQuotaClient`.

- [ ] **Step 1: Write failing tests:** `test_train_accepts_opencode_go_provider` (argparse); `test_brain_section_roundtrip` (`trainer.brain_section(BrainConfig(provider="opencode_go"))` returns `{"provider": "opencode_go", "model": "deepseek-v4.1-flash"}` and reloads equal); `test_dashboard_quota_client_for_opencode_go` returns an object whose `get_utilization()` is `None`; `test_dashboard_paused_card_says_opencode_go` (brain_status paused for `opencode_go` → title contains "OpenCode Go"); `test_status_check_reports_opencode_go_not_ready_without_key` (uses `Brain.preflight` with no key; message mentions `OPENCODE_GO_API_KEY`); `test_setup_brain_choice_opencode_go` (monkeypatch `_ask`/`getpass`: key lands in `env_vars`, model defaults to deepseek).
- [ ] **Step 2: Run** `pytest tests/test_opencode_go_surfaces.py tests/test_trainer_config.py -q`; expect FAIL.
- [ ] **Step 3: Implement** each surface, reusing `PROVIDER_LABELS`; in `cmd_usage` print "Quota gauge unavailable (OpenCode Go has no documented balance endpoint; Sellbird counts its own calls)." for `opencode_go`.
- [ ] **Step 4: Run** `pytest tests/test_opencode_go_surfaces.py tests/test_trainer_config.py tests/test_dashboard_pipeline.py tests/test_dashboard_usage_date.py tests/test_codex_cli_surface.py -q`; expect PASS.
- [ ] **Step 5: Commit** `feat: OpenCode Go in train, setup, status and dashboard`.

### Task 7: Docs, live check, full suite

**Files:**
- Create: `docs/opencode-go.md`, `tests/test_opencode_go_live.py`
- Modify: `README.md` (~66-83, ~176), `docs/configuration.md:19` and the `sellbird train` line, `CHANGELOG.md`, `CLAUDE.md` ("Can it use ChatGPT / Codex instead of Claude?"), `.env.example` (add `OPENCODE_GO_API_KEY=`)

**Interfaces:** Consumes everything above; produces documentation and a skippable live test.

- [ ] **Step 1: Write the live test** `test_live_catalog_and_completion` guarded by `pytest.mark.skipif(not (os.environ.get("SELLBIRD_OPENCODE_GO_LIVE") == "1" and os.environ.get("OPENCODE_GO_API_KEY")))`: `preflight()` passes for the default model, `run("Reply with exactly: {\"ok\":true}", timeout=60)` returns text containing `ok`, and (confirming the spec's open item) one call with `reasoning_effort="low"` succeeds. Without the env vars it is skipped.
- [ ] **Step 2: Write `docs/opencode-go.md`** following the structure of `docs/codex.md`: setup (`OPENCODE_GO_API_KEY`, YAML example with the default and a `deepseek-v4-pro` override), text-only guarantee, supported protocol and models read from the live catalog, pausing behaviour, no quota gauge, and a prominent notice that OpenCode Go is "designed for OpenCode and other coding agents" while Sellbird's traffic is outreach, so account limits or suspension are the operator's risk. Update the other docs listed.
- [ ] **Step 3: Run the full suite** `pytest -q`; expect 0 failures (live test skipped). Run it with `SELLBIRD_OPENCODE_GO_LIVE=1 pytest tests/test_opencode_go_live.py -q` only if a key is available in the environment; record the result either way.
- [ ] **Step 4: Verify privacy rule:** `git diff 8257fae...HEAD | grep -iE 'api[_-]?key *[:=] *[A-Za-z0-9]{12,}'` returns nothing, and no fixture contains a real account, email or URL beyond `opencode.ai` and `example.com`.
- [ ] **Step 5: Commit** `docs: document the OpenCode Go provider` and push.
