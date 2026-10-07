from collections.abc import Mapping

from github_intelligence.post_training.models import (
    EpisodeObservation,
    PolicyAction,
    PolicyExample,
    RewardBreakdown,
)

TOOL_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "search_issues": frozenset({"repository", "query"}),
    "get_issue": frozenset({"repository", "number"}),
    "search_pull_requests": frozenset({"repository", "query"}),
    "get_pull_request": frozenset({"repository", "number"}),
    "get_release": frozenset({"repository", "tag"}),
    "search_repository_documents": frozenset({"repository", "query"}),
}


def action_schema_is_valid(action: PolicyAction, repository: str) -> bool:
    if action.action == "finish":
        return True
    tool = action.tool
    if tool is None or tool not in TOOL_ARGUMENTS:
        return False
    required = TOOL_ARGUMENTS[tool]
    if not required.issubset(action.arguments):
        return False
    if action.arguments.get("repository") != repository:
        return False
    if "number" in required:
        number = action.arguments.get("number")
        return isinstance(number, int) and not isinstance(number, bool) and number > 0
    value_key = "tag" if "tag" in required else "query"
    value = action.arguments.get(value_key)
    return isinstance(value, str) and bool(value.strip())


def evaluate_policy_action(
    example: PolicyExample,
    action: PolicyAction,
    observation: EpisodeObservation,
) -> RewardBreakdown:
    """Compute the preregistered, programmatic pilot reward without an LLM judge."""

    gold_ids = {item.source_id for item in example.gold_evidence}
    acquired = set(observation.acquired_source_ids)
    evidence_success = float(bool(gold_ids & acquired))
    schema_validity = float(action_schema_is_valid(action, example.repository))
    selected_expected_tool = action.tool in example.acceptable_tools
    tool_success = float(observation.tool_succeeded and selected_expected_tool)
    duplicate_penalty = 0.05 if observation.duplicate_call else 0.0
    call_cost_penalty = 0.05 * min(observation.call_count / observation.call_budget, 1.0)
    invalid_identifier_penalty = 0.5 if observation.invalid_identifier else 0.0
    total = (
        0.50 * evidence_success
        + 0.20 * schema_validity
        + 0.20 * tool_success
        - duplicate_penalty
        - call_cost_penalty
        - invalid_identifier_penalty
    )
    return RewardBreakdown(
        total=total,
        evidence_success=evidence_success,
        schema_validity=schema_validity,
        tool_success=tool_success,
        duplicate_penalty=duplicate_penalty,
        call_cost_penalty=call_cost_penalty,
        invalid_identifier_penalty=invalid_identifier_penalty,
    )
