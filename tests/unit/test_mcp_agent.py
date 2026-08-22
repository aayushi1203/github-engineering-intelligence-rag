from collections.abc import AsyncIterator
from typing import Any

import pytest

from github_intelligence.agent import InvestigationAgent, InvestigationPlanner
from github_intelligence.generation import DraftAnswer, GroundedAnswerService
from github_intelligence.mcp import (
    GitHubEngineeringTools,
    ToolEvidence,
    ToolEvidenceResult,
    create_mcp_server,
)
from github_intelligence.models import DocumentType, RetrievalResult


def retrieval(identifier: str = "one", *, url: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{identifier}",
        text=f"The timeout regression was fixed by change {identifier}.",
        title=f"Fix {identifier}",
        source_url=url or f"https://github.com/acme/repo/pull/{identifier}",
        source_type=DocumentType.PULL_REQUEST,
        score=1.0,
        repository="acme/repo",
        metadata={"github_id": identifier},
    )


class FakePipeline:
    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self.results = results or []
        self.calls = 0

    def search(self, *args: object, **kwargs: object) -> list[RetrievalResult]:
        self.calls += 1
        return self.results


class CitingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, system_prompt: str, user_prompt: str) -> DraftAnswer:
        self.calls += 1
        return DraftAnswer(
            answer="The regression was fixed [S1].",
            citation_ids=["S1"],
            limitations=[],
        )


class FailingTools:
    def __init__(self) -> None:
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, object]) -> ToolEvidenceResult:
        self.calls += 1
        raise RuntimeError("GitHub unavailable")


class RepeatingTools:
    def __init__(self, items: list[ToolEvidence]) -> None:
        self.items = items
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, object]) -> ToolEvidenceResult:
        self.calls += 1
        return ToolEvidenceResult(tool_name=name, repository="acme/repo", items=self.items)


class FakeGitHubClient:
    async def get_json(
        self, url: str, params: dict[str, Any] | None = None, *, use_cache: bool = True
    ) -> Any:
        if url == "/search/issues":
            return {"items": [issue_payload(2)]}
        if "/issues/" in url:
            return issue_payload(1)
        raise AssertionError(url)

    async def paginate(
        self, url: str, params: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}


def issue_payload(number: int) -> dict[str, Any]:
    return {
        "id": number,
        "number": number,
        "title": f"Issue {number}",
        "body": "Timeout failed",
        "html_url": f"https://github.com/acme/repo/issues/{number}",
        "state": "open",
        "labels": [],
        "comments": 0,
        "user": {"login": "dev"},
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_mcp_tool_schemas() -> None:
    pipeline = FakePipeline()
    tools = GitHubEngineeringTools(FakeGitHubClient(), pipeline)  # type: ignore[arg-type]
    schemas = await create_mcp_server(tools).list_tools()
    names = {tool.name for tool in schemas}
    assert names == {
        "search_issues",
        "get_issue",
        "search_pull_requests",
        "get_pull_request",
        "get_release",
        "search_repository_documents",
    }
    get_issue = next(tool for tool in schemas if tool.name == "get_issue")
    assert set(get_issue.inputSchema["required"]) == {"repository", "number"}
    assert get_issue.outputSchema is not None


@pytest.mark.asyncio
async def test_tool_execution_returns_typed_evidence() -> None:
    tools = GitHubEngineeringTools(FakeGitHubClient(), FakePipeline())  # type: ignore[arg-type]
    found = await tools.search_issues("acme/repo", "timeout", limit=3)
    assert found.items[0].source_type == DocumentType.ISSUE
    assert str(found.items[0].source_url).endswith("/issues/2")
    inspected = await tools.get_issue("acme/repo", 1)
    assert inspected.items[0].metadata["github_id"] == "1"
    with pytest.raises(ValueError, match="positive"):
        await tools.get_issue("acme/repo", 0)


def test_investigation_planning_is_concise_and_targeted() -> None:
    plan = InvestigationPlanner().plan(
        "Which PR fixed regression #123 in release v1.2.0?", "acme/repo"
    )
    assert [action.tool_name for action in plan.actions] == ["get_pull_request", "get_release"]
    assert all("reason" not in step.lower() for step in plan.steps)


def make_agent(
    pipeline: FakePipeline,
    tools: Any,
    *,
    max_steps: int = 6,
    max_calls: int = 4,
    max_sources: int = 12,
) -> tuple[InvestigationAgent, CitingProvider]:
    provider = CitingProvider()
    service = GroundedAnswerService(pipeline, provider)  # type: ignore[arg-type]
    return (
        InvestigationAgent(
            pipeline,  # type: ignore[arg-type]
            service,
            tools,
            max_steps=max_steps,
            max_mcp_calls=max_calls,
            max_sources=max_sources,
            search_limit=2,
        ),
        provider,
    )


@pytest.mark.asyncio
async def test_evidence_deduplication_and_citation_preservation() -> None:
    first = retrieval("1")
    duplicate = ToolEvidence.from_retrieval(first)
    new = ToolEvidence.from_retrieval(retrieval("2"))
    agent, _ = make_agent(FakePipeline([first]), RepeatingTools([duplicate, new]))
    result = await agent.investigate("Which change fixed the regression?", repository="acme/repo")
    urls = [str(source.source_url) for source in result.answer.retrieved_sources]
    assert len(urls) == len(set(urls))
    assert result.answer.citations[0].source_id == "S1"
    assert str(result.answer.citations[0].source_url) in urls


@pytest.mark.asyncio
async def test_tool_failure_falls_back_to_retrieved_evidence() -> None:
    tools = FailingTools()
    agent, provider = make_agent(FakePipeline([retrieval()]), tools)
    result = await agent.investigate("Why did this bug fail?", repository="acme/repo")
    assert result.answer.citations
    assert provider.calls == 1
    assert any("failed" in limitation for limitation in result.limitations)
    assert result.tool_calls[0].succeeded is False


@pytest.mark.asyncio
async def test_step_and_tool_call_limits_stop_execution() -> None:
    tools = RepeatingTools([])
    agent, _ = make_agent(FakePipeline([retrieval()]), tools, max_steps=2, max_calls=1)
    result = await agent.investigate("Why did this bug fail?", repository="acme/repo")
    assert len(result.investigation_steps) <= 2
    assert tools.calls <= 1
    assert len(result.tool_calls) <= 1
    assert result.limitations


@pytest.mark.asyncio
async def test_zero_tool_limit_and_insufficient_evidence_do_not_loop() -> None:
    tools = RepeatingTools([])
    pipeline = FakePipeline([])
    agent, provider = make_agent(pipeline, tools, max_calls=0)
    result = await agent.investigate("Why did this bug fail?", repository="acme/repo")
    assert tools.calls == 0
    assert provider.calls == 0
    assert not result.answer.sources
    assert any("No relevant indexed" in item for item in result.limitations)
