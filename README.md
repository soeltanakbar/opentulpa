# OpenTulpa

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Self-Hosted](https://img.shields.io/badge/self--hosted-yes-green.svg)]()

OpenTulpa is a personal AI agent you run on your own server, accessible through Telegram.

It does two things most assistants don't:

- **It knows you.** It remembers facts you tell it, files you send it, preferences you express, and context from every past conversation — and uses all of that without being asked.
- **It builds its own tools.** Describe an API workflow and it writes the integration, runs it, schedules it, and saves it. You never touch code.

The longer you run it, the more personal and capable it gets.

> Two env vars. One command. A self-hosted agent that compounds over time.

---

## What It Can Do

### It Learns Who You Are

OpenTulpa remembers everything you share with it — not as a search index, but as context it actively uses when responding:

- Tell it your timezone, work schedule, or preferred tone or persona once — it applies that everywhere.
- Send it a PDF, image, or voice note — it understands and stores it for later reference.
- Mention a preference, a constraint, or a fact about your life — it factors that in unprompted from then on.
- Share a document and say "keep this in mind" — it will, across future sessions.

Early conversations are generic. Later ones feel like talking to someone who actually knows your context and acts like it.

### It Builds Its Own Integrations

Describe a workflow and OpenTulpa writes the code, runs it, schedules it, and saves it as a reusable skill — entirely from inside the chat. No dev environment. No context switching.

```text
"Pull the top 5 trending GIFs from Giphy and send me one every morning."
→ Writes the Giphy API script, schedules the job, done.

"Here's my Alpaca key. Give me a markets overview every weekday at 7am —
top movers, my portfolio delta, any earnings today."
→ Stores the key, writes the integration, registers the recurring job.

"Build me a Slack bot that posts a daily standup prompt to #engineering at 9am."
→ Writes the full Slack integration from scratch, saves it as a reusable skill.

"Register me on Moltbook." (with Browser Use connected)
→ Opens a browser, fills the form, completes the flow autonomously.

"Here's my Notion token. Summarize everything updated this week into a digest."
→ Done. Say "schedule that" and it registers the recurring job immediately.

"Write me a GitHub webhook that posts a Slack message on every failed CI run."
→ Builds both ends of the integration, inside the chat, from a single message.
```

If there's a public API or a service with documentation, OpenTulpa can integrate it without you writing a single line of code. Hand it a key or describe a service → it figures out the API → writes working code → runs it → stores it so it never rebuilds from scratch.

### Everything Else It Can Do

- **Internet research:** browse URLs, read pages, summarize findings.
- **Multimodal input:** send text, files, images, or voice notes — it handles all of them.
- **Background automation:** scheduled tasks, recurring jobs, long-running routines.
- **Artifact storage:** generated scripts and outputs are saved and reused across sessions.
- **Skills:** recurring workflows become named capabilities it maintains, loads and applies automatically.

---

## Prerequisites

- Python `3.10+`
- [`uv`](https://docs.astral.sh/uv/) installed
- Telegram bot token from `@BotFather`
- `cloudflared` installed (recommended for local webhook tunneling)

---

## Quick Start

**1. Create your Telegram bot:**
- Chat with `@BotFather` → run `/newbot` → copy the token.
- Open your new bot and press `Start`.

**2. Configure:**

```bash
cp .env.example .env
```

```bash
# .env
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
TELEGRAM_BOT_TOKEN=your_botfather_token
```

Any OpenAI-compatible endpoint works — just swap the values above.

**3. Start:**

```bash
./start.sh
```

`start.sh` will:
- Start FastAPI on `:8000`
- Launch a `cloudflared` tunnel
- Auto-register the Telegram webhook at `<public_url>/webhook/telegram`

**4. Webhook (if not using cloudflared):**

```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=https://yourdomain.com/webhook/telegram"
```

> Telegram requires a public URL. For local dev, `cloudflared` or `ngrok` both work.

**5. Start chatting.** Try:

```text
Create a daily 8:30am Gmail summary and post the top 5 action items here.
```

**Stop:** `Ctrl+C` in the terminal.

**Health checks:**
- `http://localhost:8000/healthz`
- `http://localhost:8000/agent/healthz`

---

## Skills

Skills are `SKILL.md` files the agent writes, stores, and loads on demand:

- **Scopes:** `user` (personal) and `global` (shared) — `user` always takes priority.
- **CRUD:** `skill_list`, `skill_get`, `skill_upsert`, `skill_delete`.
- **Self-authoring:** OpenTulpa can write new skills directly from chat and reuse them in future sessions without being reminded.

```text
"Create a reusable skill called Customer Follow-up Writer that takes a thread
summary and outputs 3 concise follow-up drafts in my tone."
```

---

## Configuration

**Required:**

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM routing and embeddings |
| `TELEGRAM_BOT_TOKEN` | Telegram interface |

**Optional:**

| Variable | Purpose |
|---|---|
| `BROWSER_USE_API_KEY` | Browser automation (form filling, web flows) |

**Core stack:** FastAPI · LangGraph · LangChain · mem0 · SQLite · APScheduler

| Component | Role |
|---|---|
| `mem0` | Memory layer — persists user context across sessions |
| `APScheduler` | Recurring jobs and background automation |
| `SQLite` | Local persistence, no external DB required |

**External services** (only active when you configure them):
- OpenRouter — LLM routing and embeddings
- Telegram Bot API
- Browser Use Cloud *(optional)*
- Any API you integrate yourself

**Runtime data:**
- `.opentulpa/` — memory, profiles, context
- `tulpa_stuff/` — generated scripts and artifacts *(mostly gitignored)*

---

## Safety and Privacy

- External-impact actions (writes, sends, posts) require explicit per-action approval — single-use, expiring, scoped to the requesting user only.
- No built-in telemetry or user-tracking pipeline.
- Fully open source (MIT). Self-hosted by default.
- All runtime data stays local unless you explicitly configure an external service.

---

## Project Structure

```
src/opentulpa/
├── agent/        # LangGraph runtime, graph, tool orchestration, context policy
├── api/          # App composition and internal API routes
├── interfaces/   # Telegram transport and streaming relay
├── approvals/    # Guardrail policy, broker, adapters, persistence
├── context/      # Profiles, memory events, file vault, rollups
├── tasks/        # Sandboxed execution and task runner
└── integrations/ # External service clients
tulpa_stuff/      # Generated scripts and runtime artifacts
```

Reference docs:
- [Architecture](docs/ARCHITECTURE.md)
- [External Tool Safety Checklist](docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md)

---

*If this is useful to you, consider starring the repo — it helps others find it.*
