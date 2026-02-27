"""Tool-call validation rules for graph execution."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.lc_messages import AnyMessage, ToolMessage
from opentulpa.agent.utils import extract_relative_delay_minutes as _extract_relative_delay_minutes
from opentulpa.agent.utils import is_cron_like_schedule as _is_cron_like_schedule
from opentulpa.agent.utils import latest_user_text as _latest_user_text
from opentulpa.agent.utils import looks_like_shell_command as _looks_like_shell_command

REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "tulpa_write_file": ("path", "content"),
    "tulpa_validate_file": ("path",),
    "tulpa_read_file": ("path",),
    "tulpa_run_terminal": ("command",),
    "fetch_url_content": ("url",),
    "fetch_file_content": ("url",),
    "uploaded_file_search": ("query",),
    "uploaded_file_get": ("file_id",),
    "uploaded_file_send": ("file_id",),
    "tulpa_file_send": ("path",),
    "web_image_send": ("url",),
    "uploaded_file_analyze": ("file_id",),
    "skill_get": ("name",),
    "skill_upsert": ("name", "description", "instructions"),
    "skill_delete": ("name",),
    "directive_set": ("directive",),
    "time_profile_set": ("utc_offset",),
    "browser_use_run": ("task",),
    "browser_use_task_get": ("task_id",),
    "browser_use_task_control": ("task_id",),
    "routine_list": ("customer_id",),
    "routine_create": (
        "name",
        "schedule",
        "message",
        "implementation_command",
        "customer_id",
    ),
    "routine_delete": ("routine_id", "customer_id"),
    "automation_delete": ("routine_id", "customer_id"),
    "guardrail_execute_approved_action": ("approval_id", "customer_id"),
}


def validate_tool_call(
    *,
    call_name: str,
    call_id: str,
    args: Any,
    messages: list[AnyMessage],
) -> ToolMessage | None:
    """Return a validation error ToolMessage or None when call is valid."""
    if not isinstance(args, dict):
        return ToolMessage(
            content=f"TOOL_VALIDATION_ERROR: arguments for {call_name} must be an object",
            tool_call_id=call_id,
        )

    missing = [arg for arg in REQUIRED_ARGS.get(call_name, ()) if not args.get(arg)]
    if missing:
        if call_name == "routine_create" and "implementation_command" in missing:
            return ToolMessage(
                content=(
                    "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create needs "
                    "implementation_command (a concrete shell/script command like "
                    "`python3 scripts/digest.py`) describing what will run "
                    "on each scheduled execution (the command runs with working_dir=tulpa_stuff "
                    "by default, so no tulpa_stuff/ prefix needed). Repair the call and retry."
                ),
                tool_call_id=call_id,
            )
        return ToolMessage(
            content=(
                f"TOOL_VALIDATION_ERROR: missing required argument(s) for "
                f"{call_name}: {', '.join(missing)}"
            ),
            tool_call_id=call_id,
        )

    if call_name == "tulpa_run_terminal":
        command = str(args.get("command", "")).strip()
        if not _looks_like_shell_command(command):
            return ToolMessage(
                content=(
                    "TOOL_VALIDATION_ERROR: command must be a concrete shell command "
                    "with executable + args."
                ),
                tool_call_id=call_id,
            )

    if call_name == "routine_create":
        latest_user = _latest_user_text(messages)
        schedule = str(args.get("schedule", "")).strip()
        implementation_command = str(args.get("implementation_command", "")).strip()
        if not implementation_command:
            return ToolMessage(
                content=(
                    "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create must include "
                    "a non-empty implementation_command (shell/script command) so scheduled "
                    "runs execute a concrete implementation."
                ),
                tool_call_id=call_id,
            )
        if not _looks_like_shell_command(implementation_command):
            return ToolMessage(
                content=(
                    "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command must "
                    "be a concrete shell command (executable + args), not natural language."
                ),
                tool_call_id=call_id,
            )
        delay_minutes = _extract_relative_delay_minutes(latest_user)
        if delay_minutes is not None and _is_cron_like_schedule(schedule):
            return ToolMessage(
                content=(
                    "TOOL_VALIDATION_ERROR: for one-time relative reminders, "
                    "use a local ISO datetime schedule (not cron)."
                ),
                tool_call_id=call_id,
            )

    return None
