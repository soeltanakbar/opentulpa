"""Policy decisions for external-impact action gating."""

from __future__ import annotations

from dataclasses import dataclass

from opentulpa.approvals.models import ActionIntent, GateAction

INTERNAL_ONLY_ACTIONS: set[str] = {
    "memory_search",
    "memory_add",
    "skill_list",
    "skill_get",
    "skill_upsert",
    "skill_delete",
    "directive_get",
    "directive_set",
    "directive_clear",
    "time_profile_get",
    "time_profile_set",
    "routine_list",
    "server_time",
    "guardrail_execute_approved_action",
}


@dataclass(slots=True)
class PolicyDecision:
    gate: GateAction
    reason: str


def evaluate_policy(intent: ActionIntent) -> PolicyDecision:
    if intent.action_name in INTERNAL_ONLY_ACTIONS:
        return PolicyDecision(gate="allow", reason="internal_only_action")

    if intent.llm_uncertain:
        return PolicyDecision(gate="require_approval", reason="guardrail_uncertain")

    if intent.recipient_scope == "self" and intent.impact_type not in {"purchase", "costly"}:
        return PolicyDecision(gate="allow", reason="self_target_low_impact")

    if intent.recipient_scope == "external" and intent.impact_type in {"write", "purchase", "costly"}:
        return PolicyDecision(gate="require_approval", reason="external_side_effect")

    if intent.recipient_scope == "unknown" and intent.impact_type in {"write", "purchase", "costly"}:
        return PolicyDecision(gate="require_approval", reason="unknown_destination_side_effect")

    return PolicyDecision(gate="allow", reason="default_allow")
