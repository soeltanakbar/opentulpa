# OpenTulpa Architecture

This document describes the runtime model, request flows, and extension points for OpenTulpa.

## Design goals

- Keep transport interfaces thin (Telegram is an adapter).
- Keep agent logic centralized in LangGraph runtime + tools.
- Persist user context and directives across sessions.
- Enforce external-impact safety at action execution time.

## Core components

- `src/opentulpa/api`: FastAPI composition layer and internal APIs.
- `src/opentulpa/api/routes`: concern-split route modules.
- `src/opentulpa/agent`: LangGraph runtime, graph, tools, and compaction.
- `src/opentulpa/interfaces/telegram`: Telegram transport + stream relay.
- `src/opentulpa/approvals`: external-impact guardrail policy, broker, store, adapters.
- `src/opentulpa/tasks`: background task runner + sandboxed code operations.
- `src/opentulpa/context`: user profiles, event context, file vault, rollups.
- `src/opentulpa/skills`: skill store and matching support.

## Runtime data model

- Thread state: LangGraph SQLite checkpoints (`AGENT_CHECKPOINT_DB_PATH`).
- User profile state: directive + timezone (`customer_profiles.db`).
- Event backlog: wake/task events (`context_events.db`).
- File memory: uploaded files + summaries (`file_vault.db` + file storage).
- Skills: `SKILL.md` + metadata store (`skills.db`).
- Approval lifecycle: `pending_approvals` table in approvals DB.

## Primary request flow (Telegram)

1. Telegram sends update to `/webhook/telegram`.
2. Interface layer parses user input/files/voice.
3. Agent runtime receives a user turn (`thread_id`, `customer_id`, text).
4. Runtime resolves skill context + directive + time context.
5. Graph runs with tool calls through internal APIs.
6. Approval precheck gates external-impact side effects if needed.
7. Streamed reply is emitted back to Telegram.

## Burst-message handling

Debounce/coalescing is implemented in the agent runtime (not transport):

- Multiple messages arriving before generation starts are merged into one turn.
- Messages arriving during an in-flight turn are queued for the next turn.
- Requests already merged into a previous turn are suppressed to avoid duplicate replies.

This behavior is per-thread and interface-agnostic.

## Context window policy

- Short-term context uses hysteresis: compact only when it reaches ~40k estimated tokens.
- After compaction, short-term context is reduced toward ~20k tokens.
- Older context (up to ~100k tokens per pass) is folded into a ~5k rollup injected at prompt top.

## Approval guardrail flow

1. Tool precheck calls `/internal/approvals/evaluate`.
2. Broker classifies `recipient_scope` and `impact_type`.
3. Policy returns `allow` or `require_approval`.
4. If approval is required:
   - create pending challenge
   - deliver via interface adapter (Telegram buttons / text token fallback)
5. User decision updates state (`approve`/`deny`).
6. Approved actions execute once via `/internal/approvals/execute`.

### Approval state machine

- `pending -> approved`
- `pending -> denied`
- `pending -> expired`
- `approved -> executed` (single-use)

## Background orchestration flow

1. Scheduler/Task events enqueue wake payloads.
2. Wake queue handler decides whether to notify user now or just persist context.
3. For notify-worthy events, agent drafts the user-facing message.
4. Interface delivers message to user.

## Separation of concerns

- Interface adapters do parsing/transport only.
- Agent runtime owns reasoning, turn serialization, and tool orchestration.
- Routes expose internal capability boundaries.
- Approvals own safety classification, lifecycle, and authorization checks.
- Sandbox isolates file/terminal operations for generated automation code.

## Extension points

- Add new tools in `agent/tools_registry.py`.
- Add new internal routes in `api/routes/*`.
- Add new interface adapters under `interfaces/*`.
- Add new approval adapters under `approvals/adapters/*`.
- Add skill packs via `skills` APIs.

When adding external integrations, follow:

- `docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md`

## Failure behavior

- Guardrail classification uncertainty defaults to approval-required.
- Unavailable approval store/adapters fail side-effect actions closed.
- Wake delivery failures are persisted to context backlog for recovery.
- Tool call failures propagate as explicit tool errors to the agent.
