import re

from github_intelligence.agent.models import InvestigationPlan, ToolAction

NUMBER_PATTERN = re.compile(r"#(\d+)")
TAG_PATTERN = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[\w.-]+)?\b", re.IGNORECASE)


class InvestigationPlanner:
    """Creates a bounded public plan; it does not expose model reasoning."""

    def plan(self, question: str, repository: str, *, search_limit: int = 5) -> InvestigationPlan:
        lowered = question.lower()
        actions: list[ToolAction] = []
        numbers = [int(value) for value in NUMBER_PATTERN.findall(question)]
        for number in numbers[:2]:
            tool = (
                "get_pull_request" if "pr" in lowered or "pull request" in lowered else "get_issue"
            )
            actions.append(
                ToolAction(
                    tool_name=tool,
                    arguments={"repository": repository, "number": number},
                    purpose=f"Inspect referenced GitHub artifact #{number}",
                )
            )
        tag = TAG_PATTERN.search(question)
        if tag and any(word in lowered for word in ("release", "version", "shipped")):
            value = tag.group(0)
            if not value.lower().startswith("v"):
                value = f"v{value}"
            actions.append(
                ToolAction(
                    tool_name="get_release",
                    arguments={"repository": repository, "tag": value},
                    purpose=f"Inspect release {value}",
                )
            )
        if not numbers:
            if any(word in lowered for word in ("issue", "bug", "error", "fail", "symptom", "why")):
                actions.append(
                    ToolAction(
                        tool_name="search_issues",
                        arguments={
                            "repository": repository,
                            "query": question,
                            "limit": search_limit,
                        },
                        purpose="Search related issue reports",
                    )
                )
            if any(
                word in lowered
                for word in (
                    "change",
                    "fix",
                    "regression",
                    "release",
                    "implement",
                    "added",
                    "which",
                )
            ):
                actions.append(
                    ToolAction(
                        tool_name="search_pull_requests",
                        arguments={
                            "repository": repository,
                            "query": question,
                            "limit": search_limit,
                        },
                        purpose="Search related implementation pull requests",
                    )
                )
        return InvestigationPlan(
            steps=[
                "Retrieve hybrid-reranked local evidence",
                *[action.purpose for action in actions],
                "Search indexed documents again if tools reveal useful terms",
                "Generate a citation-validated answer from deduplicated evidence",
            ],
            actions=actions,
        )
