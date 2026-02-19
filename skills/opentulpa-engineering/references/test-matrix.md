# OpenTulpa Test Matrix

## Agent Runtime (`src/opentulpa/agent/*`)

- Tool registration and required-arg validation.
- Graph routing (`agent -> validate -> tools -> agent`).
- Time-context injection behavior.
- Context compaction thresholds and rollup persistence.
- File analysis fallback behavior (text/pdf/docx/image).

## Telegram Interface (`src/opentulpa/interfaces/telegram/*`)

- Update parsing and user allowlist checks.
- Attachment extraction and ingest behavior.
- `/status`, `/setup`, key set/cancel flows.
- Streaming/upsert flow and low-signal filtering.
- Wake relay behavior and `wake_thread_id` persistence.

## API Layer (`src/opentulpa/api/app.py`)

- Health endpoints.
- Internal routes shape (`/internal/*`) for memory, files, directive, time profile, scheduler, tasks.
- Webhook guard checks (bot token and optional secret).
- Wake queue enqueue path and background handling.

## Tasks/Scheduler

- Routine creation/list/remove behavior.
- Notify-default behavior in wake payload processing.
- Event routing to Telegram vs context backlog when notification is suppressed/unavailable.

## Regression Minimum for Refactors

- Import smoke: runtime + app + telegram service.
- Route presence smoke for critical endpoints.
- Telegram `/status` handling smoke.
- Existing tests pass.
- New tests cover changed behavior.
