# OpenCode Go as a Sellbird brain provider

Date: 2026-11-10. Status: design approved in conversation; spec awaiting review.

## Goal

Let an operator run Sellbird on an OpenCode Go subscription, the way
`brain.provider: codex` runs it on a ChatGPT plan. Sellbird's agent loop
(Scout, Writer, Handler, Analyst, trainer) already lives in Python and only
calls `Brain.think()` / `Brain.think_json()`; the model returns text or JSON
and never acts. OpenCode Go therefore only has to implement that contract.

Success: with `brain: {provider: opencode_go}` in the config and
`OPENCODE_GO_API_KEY` set, `sellbird run`, `sellbird train`, `sellbird status
--check` and the dashboard work end to end, with no call to Claude or Codex.

## Decisions taken

- **Text-only brain.** No tools. The request carries no `tools` field and a
  response containing `tool_calls` is an error. Prompts hold untrusted text
  (scraped pages, prospect replies), so the model must have nothing to act with.
- **Dedicated adapter** (`sellbird/opencode_go.py`), not a generic
  OpenAI-compatible provider and not LiteLLM. The API key only ever goes to
  OpenCode's own host, fixed in code.
- **Default model: `deepseek-v4.1-flash`**, changeable with `brain.model`
  (see "Model selection").
- **Honest client identity.** The plan is documented as "designed for
  OpenCode and other coding agents that produce similar types of requests".
  Sellbird's traffic (prospecting, email drafting, reply handling) is not
  coding-agent traffic. The operator accepted that risk for their own
  account. Sellbird identifies itself as `sellbird/<version>`, sends a stable
  `x-opencode-session`, and never imitates OpenCode or a coding agent.
  `docs/opencode-go.md` states this plainly.
- **v1 protocol: `chat-completions` only.** Verified 2026-11-10 against the
  live catalog: 30 models over four protocols (`chat-completions`,
  `responses`, `messages`, `generate-content`); 19 use `chat-completions`.
  The rest are refused with a message listing the supported ones. Other
  protocols are a later version.

## Facts verified against the live service

- Base URL for chat-completions: `https://opencode.ai/inference/go/openai/v1`
  (the catalog's own `api.chat-completions.base_url`).
- Catalog: `GET <base>/models` returns each model's `id`, `type`, `api`
  (protocol and base URL), `capabilities.reasoning.effort` (the accepted
  effort values), `limit`, and `cost`.
- A request without `x-opencode-session` fails `400 MissingSessionID`.
- Success returns `choices[0].message.content` plus a separate
  `reasoning_content`, `finish_reason`, and `usage` with an
  `opencode.allowance` (what the request counted against the plan) and token
  counts.
- An unknown model returns `400` with `Upstream request failed: Model is
  unavailable.`
- Limits are monthly dollar amounts per model: 5 hours 20 %, week 50 %,
  month 100 %. No balance endpoint is documented.
- Not verified (would need exhausting the plan): the exact error shape when a
  limit is hit.

The catalog changes; the supported-model list is read at runtime, never
hard-coded.

## Components

### `sellbird/opencode_go.py` (new)

- `OpenCodeGoError(Exception)` with `kind` in `auth | setup | usage_limit |
  transient | paused` and `retryable`, mirroring `CodexError` so `Brain`
  reuses its pause and retry logic unchanged.
- `OpenCodeGoResult`: `text`, `model`, `usage` (the same keys
  `record_usage_event` already takes), `duration_ms`, `session_id`,
  `allowance`.
- `OpenCodeGoClient(model, reasoning_effort)`:
  - `list_models()` → catalog dict, cached for the process with a short TTL.
  - `preflight()` → checks, in order: key present; `GET /models` succeeds
    (401/403 → `auth`); the configured model is listed (else `setup`, with
    the supported ids); its protocol is `chat-completions` (else `setup`);
    `reasoning_effort`, if set, is in the model's effort list. Sends no
    prompt and spends no allowance.
  - `run(prompt, timeout, skip_if)` → one `POST /chat/completions` with a
    single `user` message and `max_tokens` large enough to leave room for
    reasoning. Headers: `Authorization: Bearer <key>`,
    `User-Agent: sellbird/<version>`, `x-opencode-session: <id>`.
  - Guards: URL fixed in code; redirects disabled; no `tools`/`tool_choice`;
    reject any `tool_calls`; reject `finish_reason` other than `stop`, except
    `length`, which is retried once with a larger `max_tokens` before failing
    as `transient`; only `content` is returned (never `reasoning_content`);
    the key never appears in logs or error text (`logsafety.redact`).
  - Error mapping: 401/403 → `auth`; 400 unavailable-model or missing-key →
    `setup`; 429 or a body mentioning a usage/limit/quota condition →
    `usage_limit`; 5xx, timeouts, connection errors → `transient`.
  - Session id: a per-process random id combined with the call's agent
    label, so one conversation stream keeps one id.
- `OpenCodeGoQuotaClient`: `get_utilization()` returns `None` in v1 (no
  documented balance endpoint), so the budget gate falls back to Sellbird's
  own call counter, and the dashboard shows "Quota unavailable". If a
  `usage_limit` error occurs, `Brain` pauses calls for an hour, as with Codex.

### `sellbird/config.py`

- `BRAIN_PROVIDERS = ("claude", "codex", "opencode_go")`.
- A model-level validator on `BrainConfig` (it needs `provider` and `model`
  together):
  - `opencode_go` with an empty `model` → `deepseek-v4.1-flash`
    (`OPENCODE_GO_DEFAULT_MODEL`).
  - `reasoning_effort` accepts, for `opencode_go`, the wider set
    `none, minimal, low, medium, high, xhigh, max`; the exact per-model list
    is checked at preflight against the catalog. Claude and Codex keep their
    current rules.
- The existing model-name regex already fits OpenCode ids.

### `sellbird/brain.py`

- `PROVIDER_LABELS["opencode_go"] = "OpenCode Go"`.
- `__init__` builds `OpenCodeGoClient` and `OpenCodeGoQuotaClient`.
- `_think_codex` becomes a shared `_think_adapter(client, error_cls, source)`
  used by Codex and OpenCode Go; Claude's path is untouched. The persistent
  failure kinds (`setup`, `auth`) pause the Brain; `isolation` stays Codex-only.
- `preflight`, `_fingerprint` (the key's presence and the model) and the usage
  recording gain an `opencode_go` branch. Usage rows use
  `provider="opencode_go"`, `source="opencode_go_http"`, and record
  `opencode.allowance` as `cost_usd`.
- Never falls back to another provider.

### `sellbird/state.py`

- `BRAIN_PROVIDERS` and `_USAGE_LOG_COLUMN` gain `opencode_go`.
- Additive migration: `ALTER TABLE usage_log ADD COLUMN opencode_go_calls
  INTEGER DEFAULT 0`, written like the existing `codex_calls` migration;
  `usage_today_by_provider` reports it.

### CLI, setup wizard, dashboard

- `sellbird train --provider opencode_go [--model ID]`.
- `sellbird setup`: new choice; asks for the key without echoing it, saves it
  to `.env` as `OPENCODE_GO_API_KEY`, lists the supported models, suggests
  the default, and runs `preflight`.
- `sellbird status` / `status --check` and the dashboard Usage and "brain
  paused" cards show "OpenCode Go".
- The trainer already carries the `brain:` block through retraining.

## Model selection

Order: `brain.model` in the config (or `--model`) wins; otherwise
`deepseek-v4.1-flash`. The model is validated against the live catalog at
preflight; an unknown or unsupported model pauses the cycle with the list of
valid ids, never a silent substitution. Changing models is editing one line:

```yaml
brain:
  provider: opencode_go
  model: deepseek-v4-pro      # optional; default deepseek-v4.1-flash
  reasoning_effort: high      # optional; must be one the model lists
```

## Error handling

Same contract as Codex: persistent problems (no key, bad key, unsupported
model or protocol) pause whole cycles with a reason in the log,
`sellbird status` and the dashboard; transient ones skip a call and retry
with the existing bounded backoff; a usage-limit error pauses calls for an
hour. A pause recorded for one provider is ignored when another is configured.

## Testing

Offline, with `httpx.MockTransport` and fixtures built from fictional data
(no real accounts, per CONTRIBUTING.md). New `tests/test_opencode_go.py`:

- success path returns only `content`, records allowance and tokens;
- headers sent: bearer, `sellbird/<version>` user-agent, stable session id;
  no `tools` field in the body;
- 401, 403, 429/limit body, 5xx, timeout, connection error each map to the
  right `kind`; retries are bounded;
- `tool_calls` in a response is rejected; a redirect is not followed;
- the key never appears in logs or exceptions;
- `length` finish retried once, then fails;
- preflight: missing key, unknown model, non-chat-completions model, bad
  effort, success with no prompt sent;
- default model applied when `model` is empty; explicit model respected.

Extend `tests/test_brain_provider.py` (opencode_go never starts `claude` or
`codex`; pause stickiness; usage-limit pause; budget uses the call counter),
the config tests (validator, effort sets per provider), the state tests
(migration on an existing database, per-provider counts) and
`tests/test_trainer_config.py`. A live check, skipped unless
`SELLBIRD_OPENCODE_GO_LIVE=1` and a key are present, calls `list_models` and
one short completion.

## Documentation

New `docs/opencode-go.md` (setup, model choice, limits, the intended-use
notice, troubleshooting); updates to README, `docs/configuration.md`,
`CHANGELOG.md` and the "Can it use ChatGPT / Codex instead of Claude?"
section of `CLAUDE.md`.

## Out of scope (v1)

`responses`, `messages` and `generate-content` models; tool use of any kind;
reading plan balance (no documented endpoint); automatic model switching;
fallback to another provider; running the `opencode` CLI.

## Open items to confirm during implementation

- Exact body of the usage-limit error (parsed defensively; the mock tests pin
  the assumed shapes).
- Whether `reasoning_effort` is accepted as that request field on
  chat-completions for each model (the catalog lists efforts, not the
  parameter name); confirm with one live call per effort style before
  documenting it.
- A sensible `max_tokens` ceiling per model from `limit.output`.
