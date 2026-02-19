# OpenTulpa

Background-capable AI agent built with **LangGraph** (OpenRouter-backed model) and **mem0** for persistent memory.

## Architecture

- **FastAPI** (port `8000`): health endpoints, internal APIs, Telegram webhook.
- **LangGraph runtime** (in-process): tool-calling agent with SQLite checkpoint persistence.
- **mem0**: persistent user memory (add/search).
- **Scheduler + Task services**: background orchestration, wake queue, and artifacts.

## Prerequisites

- Python 3.10+
- `OPENROUTER_API_KEY` (required for agent + memory model calls)
- Optional: `TELEGRAM_BOT_TOKEN`
- Optional: `SLACK_BOT_TOKEN`

## Run

```bash
./start.sh
```

- API: `http://localhost:8000`
- Health: `http://localhost:8000/healthz`
- Agent health: `http://localhost:8000/agent/healthz`

If `OPENROUTER_API_KEY` is missing, FastAPI still starts for setup flows, but chat remains disabled until the key is set and the app is restarted.

## Observability

Telemetry/tracing is configured directly in OpenRouter (built-in provider traces in the OpenRouter dashboard).

- This repository does not run a local telemetry stack.
- No local OpenTelemetry/OpenLIT instrumentation is required for normal operation.

## Environment

See `.env.example` for full configuration. Core variables:

- `OPENROUTER_API_KEY`
- `LLM_MODEL` (default `gemini-3-flash-preview`)
- `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`)
- `OPENROUTER_EMBEDDING_MODEL` (default `openai/text-embedding-3-small`)
- `MEM0_QDRANT_PATH` (default `.opentulpa/qdrant`)
- `MEM0_QDRANT_ON_DISK` (default `true`)
- `AGENT_CHECKPOINT_DB_PATH` (default `.opentulpa/langgraph_checkpoints.sqlite`)
- `AGENT_RECURSION_LIMIT` (default `30`)
- `AGENT_CONTEXT_TOKEN_LIMIT` (default `250000`, estimated tokens before compaction)
- `AGENT_CONTEXT_ROLLUP_TOKENS` (default `100000`, oldest estimated tokens summarized per compaction)

## Telegram mode

- Webhook endpoint: `POST /webhook/telegram`
- One Telegram chat maps to one persistent LangGraph thread.
- Stable customer id per user: `telegram_<user_id>`
- Telegram is just the interface; preference/directive persistence is handled in the LangGraph layer.

Commands:

- `/start` or `/help`
- `/status`
- `/setup`
- `/set KEY VALUE` or `/setenv KEY VALUE`
- `/cancel`

You can also set long-lived behavior preferences in plain language during normal chat, for example:
- "From now on, keep answers very concise."
- "When writing code, prefer small pure functions and type hints."
- "Forget the previous writing style preferences."

The agent stores one active persistent directive per user and overwrites it when you provide a new one.

For links and documents, you can send a URL directly (HTML, PDF, DOCX, images). The LangGraph layer
uses a dedicated content-fetch tool to read exact links, and falls back to web search only when needed.

Scheduled routines default to direct user notification (`notify_user=true`). To suppress alerts,
explicitly ask for no notification (for example: "run this silently" or "don't alert me").

Each turn injects live server time + best-known user local time/UTC offset into the model context.
If the user timezone is unknown, server timezone is used as a fallback until the agent stores a user offset.
Customer metadata (directive + timezone/offset) is stored in a unified customer profile store.

## Project layout

- `src/opentulpa/agent` — LangGraph runtime (`runtime.py`), state models, and utilities
- `src/opentulpa/api` — FastAPI app and internal APIs
- `src/opentulpa/interfaces/telegram` — Telegram client/formatter/chat service
- `src/opentulpa/integrations` — Slack and web-search integrations
- `src/opentulpa/tasks` — sandbox, task service, wake queue
- `tulpa_stuff` — dynamic integration artifacts
