import json

from pydantic import ValidationError

from github_intelligence.post_training.models import PolicyAction, PolicyExample

POLICY_SYSTEM_PROMPT = """You are a bounded GitHub investigation policy.
Choose exactly one next action. Return JSON only, matching this schema:
{"action":"tool_call","tool":"<tool>","arguments":{...},"reason_code":"<short_code>"}
Available tools: search_issues, get_issue, search_pull_requests, get_pull_request,
get_release, search_repository_documents. Never invent issue numbers, pull request
numbers, or release tags. Use search when the question does not explicitly provide an ID."""


def policy_prompt(example: PolicyExample) -> str:
    return (
        f"Repository: {example.repository}\n"
        f"Question: {example.question}\n"
        "No evidence has been acquired yet. Select the next action."
    )


def oracle_action(example: PolicyExample) -> PolicyAction:
    tool = example.acceptable_tools[0]
    return PolicyAction(
        action="tool_call",
        tool=tool,
        arguments={"repository": example.repository, "query": example.question},
        reason_code="locate_evidence",
    )


def sft_record(example: PolicyExample) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": policy_prompt(example)},
            {
                "role": "assistant",
                "content": oracle_action(example).model_dump_json(),
            },
        ],
        "example_id": example.example_id,
    }


def parse_policy_action(completion: str) -> PolicyAction | None:
    text = completion.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        payload = json.loads(text.strip())
        return PolicyAction.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


def completion_reward(example: PolicyExample, completion: str) -> float:
    """One-step GRPO reward: valid JSON, valid arguments, and correct tool routing."""

    action = parse_policy_action(completion)
    if action is None:
        return -0.5
    from github_intelligence.post_training.reward import action_schema_is_valid

    schema = action_schema_is_valid(action, example.repository)
    correct_tool = action.tool in example.acceptable_tools
    return 0.4 * float(schema) + 0.6 * float(correct_tool)

