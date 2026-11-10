# Using an OpenCode Go plan instead (OpenCode Go)

Claude is the default and needs no change. If you'd rather spend an [OpenCode Go](https://opencode.ai) subscription, Sellbird can think through its API instead: still a subscription with monthly allowances, never per-token billing.

```bash
# 1. Subscribe to OpenCode Go and copy the API key issued for the plan
#    (not a pay-as-you-go key: that one bills your balance instead of the plan).
# 2. Put it in .env:
OPENCODE_GO_API_KEY=...
```

Then either pick **opencode_go** in `sellbird setup` (it asks for the key without showing it and checks it), run `sellbird train <url> --provider opencode_go`, or add to your config:

```yaml
brain:
  provider: opencode_go
  model: deepseek-v4-pro      # optional; the default is deepseek-v4.1-flash
  reasoning_effort: high      # optional; must be one the model lists
```

## Read this first: what the plan is for

OpenCode's own page says Go is "designed for OpenCode and other coding agents that produce similar types of requests", and that traffic is monitored for abuse. Sellbird is not a coding agent: it prospects, drafts email and handles replies. Using the plan for that is outside what it describes, so OpenCode could limit or suspend it. That risk is yours to weigh before you choose this provider.

Sellbird does not try to hide what it is. It sends `User-Agent: sellbird/<version>` and a stable `x-opencode-session`, as the service asks of clients, and never imitates OpenCode or a coding agent. If you'd rather not take the risk, keep Claude or Codex.

## What to know

- **It's one or the other.** Sellbird uses exactly the provider you configure. If OpenCode Go fails, the call fails with a clear log message; it never falls back to Claude, Codex or an API-billed route.
- **No tools, ever.** Prompts contain untrusted text (prospect replies, scraped pages), so the model must have nothing to act with. Every request is one `user` message with no `tools`, `tool_choice` or `stream` field, and a reply that asks for a tool call is refused. Only the answer text is used: the model's reasoning text is never returned.
- **Your key only goes to OpenCode.** The addresses are fixed in the code (`https://opencode.ai/inference/go/...`), redirects are not followed, and the key never appears in logs or error messages. Keep it in `.env` (`sellbird setup` writes the file with owner-only permissions). A key in `.env` takes precedence over one in the environment, so fixing the file is enough; a key set only in the environment needs a restart to change.
- **The default model is DeepSeek.** `deepseek-v4.1-flash` unless you set `brain.model`. Any model from your catalog that uses the `chat-completions` protocol works (DeepSeek, GLM, Kimi, MiniMax and others). Models on the `responses`, `messages` or `generate-content` protocols aren't supported yet and are refused with the list of ones that are. The catalog changes, so Sellbird reads it live rather than keeping a list: an unknown model name pauses Sellbird and tells you which names are valid. It never swaps in another model for you.
- **Reasoning effort.** Optional. Each model lists the efforts it accepts (for example `low`, `high`, `max`); Sellbird checks yours against that list before sending anything.
- **Limits and quota.** Plan limits are monthly dollar amounts per model, with a 5-hour window at 20% of the month and a weekly window at 50%. OpenCode has no documented balance endpoint, so Sellbird can't show a gauge: the dashboard and `sellbird usage` say "Quota unavailable", and the budget falls back to Sellbird's own daily call count (`usage.max_daily_claude_percent` applies to that count). Each reply reports what it counted against your plan, and Sellbird stores that amount with the call in its usage log. If a call fails on the plan's usage limit, Sellbird stops OpenCode Go calls for an hour. Check your remaining allowance in the OpenCode console.
- **When OpenCode Go can't be used, Sellbird pauses rather than guessing.** Before each heartbeat Sellbird reads your model list with the key and checks that your model and effort are usable. A missing or rejected key, an unusable model or a usage limit pauses the whole cycle: no reply is read or marked handled, nothing is drafted or sent. The log, `sellbird status` and the dashboard say "OpenCode Go paused" and why. A replaced key in `.env` is noticed on the next heartbeat; a changed `brain.model` needs a restart; other fixes are picked up within 30 minutes. `sellbird status --check` runs the same check on demand and exits non-zero if it fails. Brief failures (a timeout or a server error) just skip one call.
- **Requirements.** Python 3.11 and an OpenCode Go key. No extra CLI and no Node.js. It works on every platform Sellbird supports, Docker included.

## Troubleshooting

| What you see | What to do |
|---|---|
| "No OpenCode Go API key" | Add `OPENCODE_GO_API_KEY` to `.env` and restart. |
| "rejected the API key" | The key is wrong, or your Go subscription isn't active. Use the key issued for the plan, not a pay-as-you-go one. |
| "is not in your OpenCode Go catalog" | `brain.model` isn't a model your plan lists. The message names the ones Sellbird can use. |
| "uses the 'responses' protocol" | That model isn't supported yet. Choose one from the list in the message. |
| "doesn't offer reasoning_effort" | Pick one of the efforts named in the message, or leave `reasoning_effort` blank. |
| "usage limit reached for this plan" | You're out of allowance for the window. Sellbird resumes after an hour; check the OpenCode console. |

To check the live service from your machine with your own key: `SELLBIRD_OPENCODE_GO_LIVE=1 pytest tests/test_opencode_go_live.py`.
