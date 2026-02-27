# OpenTulpa Architecture

Last updated: February 27, 2026

This document describes the repository structure after the current refactor, including layer boundaries, runtime decomposition, and code conventions.

## Design goals

- Keep transport, orchestration, and agent reasoning separate.
- Keep API routes transport-only (validation + delegation).
- Keep agent runtime behavior modular and testable.
- Keep safety gating at execution boundaries for external-impact actions.
- Keep persistent context (profiles, memory, rollups, files, approvals) durable across sessions.

## Dependency direction

Allowed dependency flow:

1. `api/routes` -> `application` -> `domain/context/services`
2. `interfaces/*` -> `application` and/or `agent`
3. `agent` -> `context`, `policy`, `approvals`, `integrations` (via internal APIs/tools)

Disallowed by convention:

- Business logic in route handlers.
- FastAPI types imported into `application/*`.
- Circular imports through package-level aggregators.

## Repository map

- `src/opentulpa/api`
  - App composition and route registration.
  - Request parsing helpers in `api/errors.py`.
  - Request/query DTOs in `api/schemas/*` (Pydantic).
- `src/opentulpa/application`
  - Orchestrators per domain (`*_orchestrator.py`).
  - Shared `ApplicationResult` in `application/contracts.py`.
  - `application/__init__.py` intentionally minimal to avoid import cycles.
- `src/opentulpa/domain`
  - Core domain contracts (for example conversation turn contracts).
- `src/opentulpa/agent`
  - LangGraph runtime and behavior modules.
  - Graph composition in `graph_builder.py`.
  - Node routing in `graph_routes.py`.
  - Node implementations in focused modules:
    - `graph_node_agent.py`
    - `graph_node_validate.py`
    - `graph_node_tools.py`
    - `graph_node_claim_check.py`
    - `graph_node_limits.py`
  - `graph_nodes.py` is now composition/export only.
  - Runtime decomposition:
    - `runtime.py` (facade + composition)
    - `runtime_turns.py`
    - `runtime_lifecycle.py`
    - `runtime_classification.py`
    - `runtime_facade.py`
    - plus focused helpers (`runtime_*` modules).
  - Tools decomposition:
    - Composition in `tools_registry.py`
    - Shared helper utilities in `tools_registry_support.py`
    - Domain tool modules in `agent/tools/*`.
- `src/opentulpa/interfaces/telegram`
  - Telegram transport split by concern:
    - `chat_service.py` (chat ingress orchestration)
    - `chat_commands.py` (command/control branch handling)
    - `relay_streaming.py` (stream delivery behavior)
    - `relay_events.py` (wake/task event relay behavior)
    - `relay.py` (thin composition facade)
    - plus `client.py`, `attachments.py`, `session_state.py`, etc.
- `src/opentulpa/approvals`, `src/opentulpa/policy`
  - Approval lifecycle + policy evaluation.
- `src/opentulpa/context`
  - Persistent context services (profiles, rollups, files, links, events).
- `src/opentulpa/tasks`, `src/opentulpa/scheduler`, `src/opentulpa/skills`
  - Task execution, scheduling, skill storage/services.

## Primary runtime flows

### Telegram message flow

1. `POST /webhook/telegram` receives update.
2. `TelegramWebhookOrchestrator` handles callback/message branching.
3. Message path delegates to `TelegramChatService.handle_update`.
4. Chat path parses text/attachments, resolves session/thread, and calls:
   - `runtime.astream_text(...)` for streaming Telegram responses, or
   - `runtime.ainvoke_text(...)` fallback path.
5. Deferred approval challenges are flushed after response delivery.

### Internal chat flow

1. `POST /internal/chat` validates request DTO.
2. Route delegates to `TurnOrchestrator`.
3. Runtime executes turn.
4. Route returns normalized response payload.

### Approval flow

1. Tool/action calls are evaluated through approval policy/broker.
2. Pending approvals are stored durably.
3. Telegram callback approve/deny updates approval state.
4. Approved actions execute through `ApprovalExecutionOrchestrator`.
5. Result summary is sent back to user.

### Wake flow

1. Task/scheduler/approval events produce wake payloads.
2. `WakeOrchestrator` decides notify vs backlog.
3. Notify events are relayed through Telegram + agent drafting.
4. Non-notify events are persisted to context backlog.

## LangGraph state machine

Current node graph:

1. `agent`
2. `validate_tools` (if tool calls exist)
3. `tools`
4. `claim_check`
5. loop to `agent` or `END` based on route predicates

Notes:

- Tool-call validation occurs before tool execution.
- Guardrail approval logic is enforced at execution boundary tools.
- Claim-check retries with bounded backoff for empty/mismatched outcomes.

## Current contracts and typing

- API transport contracts: Pydantic request/query DTOs in `api/schemas/*`.
- Agent classifier/result contracts: Pydantic models in `agent/result_models.py`.
- Application contracts: `ApplicationResult[TPayload]`.
- Current orchestrator payloads are standardized as `dict[str, object]`.

## Runtime data stores

- LangGraph checkpoints: `.opentulpa/langgraph_checkpoints.sqlite`
- Approvals: `.opentulpa/pending_approvals.db`
- Context events: `.opentulpa/context_events.db`
- Customer profiles: `.opentulpa/customer_profiles.db`
- Thread rollups: `.opentulpa/thread_rollups.db`
- Link aliases: `.opentulpa/link_aliases.db`
- Skills: `.opentulpa/skills.db`
- File vault: `.opentulpa/file_vault.db` + file storage
- Tasks/wake queue: `.opentulpa/tasks.db`, `.opentulpa/wake_events.db`

## Codebase conventions

### Route conventions

- Parse request bodies via `parse_request_model(...)`.
- Parse query params via `parse_query_model(...)`.
- Delegate to an orchestrator; do not embed domain logic in route functions.
- Return orchestrator payload or `JSONResponse` for non-200 results.

### Application conventions

- One orchestrator per domain boundary (`*_orchestrator.py`).
- No FastAPI/transport imports in application layer.
- Return `ApplicationResult`.
- Keep side effects explicit and isolated.

### Agent conventions

- `runtime.py` is a facade, not a monolith.
- Put new runtime behavior in focused `runtime_*` modules.
- Keep `tools_registry.py` composition-only; place tool logic in `agent/tools/*`.
- Keep graph node logic in `graph_node_*` modules; `graph_nodes.py` is an export surface.

### Telegram conventions

- Keep transport parsing in `client.py` and `chat_service.py`.
- Keep command handling in `chat_commands.py`.
- Keep streaming behavior in `relay_streaming.py`.
- Keep event relay behavior in `relay_events.py`.

### Purity and maintenance

- Avoid compatibility wrappers unless required for an active migration window.
- Prefer explicit typed contracts over `Any` for new boundaries.
- Preserve strict layer boundaries when adding features.

## Security and boundary policy

- `/webhook/*` is public ingress.
- `/internal/*` is server-local-only surface.
- `/webhook/telegram` requires `x-telegram-bot-api-secret-token`.
- External-impact actions require approval unless explicitly safe/allowed by policy.

## Observability

- Behavior log: `.opentulpa/logs/agent_behavior.jsonl`
- Controlled by:
  - `AGENT_BEHAVIOR_LOG_ENABLED`
  - `AGENT_BEHAVIOR_LOG_PATH`

## How to extend safely

### Add a new internal endpoint

1. Add request/query DTO in `api/schemas/`.
2. Add/extend orchestrator in `application/`.
3. Add thin route in `api/routes/`.
4. Add/adjust tests for route + orchestrator.

### Add a new tool

1. Implement domain tool function(s) in `agent/tools/<domain>_tools.py`.
2. Wire it in `tools_registry.py` composition.
3. Add guardrail/approval handling if external-impacting.
4. Add tool execution and integration tests.

### Add a new Telegram behavior

1. Place logic in the appropriate Telegram module (`chat_commands`, `relay_streaming`, or `relay_events`).
2. Keep `chat_service.py` and `relay.py` as orchestration/composition boundaries.
3. Add focused tests per module.

## Quality gates

Run before merging refactor changes:

1. `uv run ruff check`
2. `uv run pytest -q`

For external integrations and action safety, see `docs/EXTERNAL_TOOL_SAFETY_CHECKLIST.md`.
