# OpenTulpa

OpenTulpa is a safe personal agent backend for developers.
It combines chat interfaces (Telegram today), tool-calling (LangGraph), memory (mem0), and automation (scheduler/tasks) so one assistant can reason, execute, and persist context across sessions while gating external-impact actions through approvals.

Developer-first personal agent runtime built on LangGraph, OpenRouter, and mem0.

OpenTulpa is optimized for building a reliable assistant you can actually modify: clear Python code, explicit action-time guardrails, persistent user context, and scheduler/task orchestration.

## Why OpenTulpa

- Pragmatic Python stack: FastAPI + LangGraph + SQLite + mem0.
- Strong action safety model: external-impact operations require approval.
- Persistent behavior control: per-user directive + timezone + memory.
- Built for iterative automation: routines, wake queue, task artifacts, and tool calls.
- Clean modular API surface: route modules split by concern.

## OpenTulpa vs OpenClaw (dev view)

OpenClaw (from its README/docs) is broad and platform-heavy: many channels, apps, and ecosystem surface.

OpenTulpa takes a narrower stance:

- Scope: smaller surface area, easier to reason about and fork quickly.
- Runtime model: in-process LangGraph with explicit tool APIs and checkpoints.
- Safety posture: approval broker wired into execution path for external-impact effects.
- Developer ergonomics: modular routes, explicit services, and straightforward extension points.

If you want maximal channel/platform breadth, OpenClaw has more out-of-the-box surface.
If you want a simpler codebase with explicit control over agent behavior and safety, OpenTulpa is usually easier to evolve.

## Strengths

- Action-time guardrails, not brittle keyword matching.
- Deterministic approval lifecycle (`pending -> approved/denied/expired -> executed`).
- Per-user state model (directive overwrite, timezone profile, long-term memory).
- Telegram-first UX with streaming, files, links, images, voice transcription.
- Good refactorability: API routes separated into focused modules.

## Capabilities

- Conversational assistant with persistent per-user identity (`telegram_<user_id>`), memory, directive, and timezone context.
- Tool-driven web workflows: web search, exact link content retrieval, and browser automation tasks (when configured).
- File intelligence: ingest/search/get/analyze uploaded files (PDF, DOCX, images, text, audio/voice).
- Voice handling: transcribes Telegram voice notes and injects transcript into turn context.
- Messaging/media actions: send stored files and web-fetched images back to user sessions.
- Automation: create/list/delete scheduled routines, wake events, and task orchestration with artifacts.
- Code/task execution path: sandboxed terminal/file operations for controlled automation tasks.

## Use Cases

- Background integration worker: register and post to Moltbook through a scheduled/background workflow while preserving a consistent assistant personality and tone.
- Market watcher: write scripts + custom parsers that run every few hours to fetch current stock data, generate analysis, and send compact updates.
- API workflow builder: add API keys through chat setup, connect external APIs, and compose custom multi-step workflows by chatting with OpenTulpa.
- Growth autopilot: create an AgentsMail identity, sign up to relevant resources/social networks, and publish posts based on trend signals.
- Personal CRM + ops copilot: track contacts, reminders, follow-ups, and next actions.
- Sales/research assistant: discover targets, enrich data, and draft outreach variants.
- Inbox and notification triage: summarize noisy streams and escalate only important items.
- Incident/runbook assistant: detect failures, collect diagnostics, suggest fixes, and notify.

## Skill System

OpenTulpa supports reusable skills as first-class runtime assets:

- Skill scopes: `user` and `global`, with user-overrides-global precedence.
- Storage: each skill is persisted as `SKILL.md` (+ optional supporting files) and indexed in SQLite.
- Runtime behavior: agent lists available skills, selects relevant ones via model reasoning, loads matched skill content, then executes with tool calls.
- CRUD tooling: `skill_list`, `skill_get`, `skill_upsert`, `skill_delete`.
- Default behavior: a built-in skill-creator baseline is provisioned so the agent can help users define new reusable workflows.

## Quick Start (Telegram in Minutes)

1. Create a Telegram bot with BotFather:
   - Open Telegram, chat with `@BotFather`
   - Run `/newbot`
   - Copy the bot token
2. Set the minimum env vars in `.env`:

```bash
OPENROUTER_API_KEY=your_openrouter_key
TELEGRAM_BOT_TOKEN=your_botfather_token
```

3. Start OpenTulpa:

```bash
./start.sh
```

4. Ensure your Telegram webhook points to `/webhook/telegram` (via your tunnel/public URL setup), then send `/start` to your bot.

That is enough to get a working personal agent.

Health checks:

- `http://localhost:8000/healthz`
- `http://localhost:8000/agent/healthz`

## Core Config

See `.env.example`. Most important:

- `OPENROUTER_API_KEY`
- `LLM_MODEL` (default `gemini-3-flash-preview`)
- `OPENROUTER_BASE_URL`
- `OPENROUTER_EMBEDDING_MODEL`
- `AGENT_CONTEXT_TOKEN_LIMIT` (default `40000`, short-term high-watermark)
- `AGENT_CONTEXT_RECENT_TOKENS` (default `20000`, short-term low-watermark target)
- `AGENT_CONTEXT_ROLLUP_TOKENS` (default `5000`)
- `AGENT_CONTEXT_COMPACTION_SOURCE_TOKENS` (default `100000`)
- `BROWSER_USE_API_KEY` / `BROWSER_USE_BASE_URL` (optional browser automation)

## Safety Model (External Impact)

Approval checks happen at tool execution precheck.

- Self-targeted replies/files to the current user session: allowed.
- External recipient or unknown scope + side effects/cost: approval required.
- Unknown/uncertain classification: fail closed to approval.
- Approvals are origin-user-only, single-use, and expire (default 10 min).

Failure behavior:

- Guardrail classifier errors default to approval-required.
- Approval store/adapter failure blocks side-effect action.
- Wake relay failures are captured to context events for recovery.

Use this before adding integrations:

- [`docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md`](docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md)

## Architecture

- `src/opentulpa/agent`: LangGraph runtime, graph, tools, context handling.
- `src/opentulpa/api`: app composition.
- `src/opentulpa/api/routes`: focused route modules (`health`, `files`, `skills`, `scheduler`, `tasks`, etc.).
- `src/opentulpa/approvals`: policy, broker, adapters, persistence.
- `src/opentulpa/interfaces/telegram`: transport client + chat orchestration.
- `src/opentulpa/integrations`: web search and Browser Use clients.
- `src/opentulpa/tasks`: sandboxed execution, task service, wake queue.
- `tulpa_stuff`: dynamic/generated integration artifacts.
