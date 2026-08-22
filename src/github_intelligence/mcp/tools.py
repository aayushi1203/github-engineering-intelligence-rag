from typing import Any

from github_intelligence.github import GitHubClient, GitHubError
from github_intelligence.ingestion.normalizers import (
    normalize_comment,
    normalize_issue,
    normalize_pull_request,
    normalize_release,
)
from github_intelligence.mcp.models import ToolEvidence, ToolEvidenceResult
from github_intelligence.retrieval import RetrievalMode, RetrievalPipeline


class GitHubEngineeringTools:
    """Shared implementation used directly by the agent and exposed through MCP."""

    def __init__(self, client: GitHubClient, pipeline: RetrievalPipeline) -> None:
        self.client = client
        self.pipeline = pipeline

    async def search_issues(
        self, repository: str, query: str, limit: int = 5
    ) -> ToolEvidenceResult:
        items = await self._search(repository, query, "issue", limit)
        documents = [normalize_issue(repository, item) for item in items]
        return self._documents("search_issues", repository, documents)

    async def get_issue(self, repository: str, number: int) -> ToolEvidenceResult:
        self._positive_number(number)
        item = await self.client.get_json(f"/repos/{repository}/issues/{number}")
        if "pull_request" in item:
            raise GitHubError(f"#{number} is a pull request, not an issue")
        documents = [normalize_issue(repository, item)]
        async for comment in self.client.paginate(f"/repos/{repository}/issues/{number}/comments"):
            documents.append(normalize_comment(repository, comment, number, pull_request=False))
        return self._documents("get_issue", repository, documents)

    async def search_pull_requests(
        self, repository: str, query: str, limit: int = 5
    ) -> ToolEvidenceResult:
        items = await self._search(repository, query, "pr", limit)
        documents = [normalize_pull_request(repository, item) for item in items]
        return self._documents("search_pull_requests", repository, documents)

    async def get_pull_request(self, repository: str, number: int) -> ToolEvidenceResult:
        self._positive_number(number)
        item = await self.client.get_json(f"/repos/{repository}/pulls/{number}")
        documents = [normalize_pull_request(repository, item)]
        async for comment in self.client.paginate(f"/repos/{repository}/issues/{number}/comments"):
            documents.append(normalize_comment(repository, comment, number, pull_request=True))
        async for comment in self.client.paginate(f"/repos/{repository}/pulls/{number}/comments"):
            documents.append(
                normalize_comment(repository, comment, number, pull_request=True, review=True)
            )
        return self._documents("get_pull_request", repository, documents)

    async def get_release(self, repository: str, tag: str) -> ToolEvidenceResult:
        if not tag.strip():
            raise ValueError("release tag must not be blank")
        item = await self.client.get_json(f"/repos/{repository}/releases/tags/{tag}")
        return self._documents("get_release", repository, [normalize_release(repository, item)])

    async def search_repository_documents(
        self, repository: str, query: str, limit: int = 5
    ) -> ToolEvidenceResult:
        results = self.pipeline.search(
            query, mode=RetrievalMode.HYBRID_RERANK, top_k=limit, repository=repository
        )
        items = [ToolEvidence.from_retrieval(result) for result in results]
        return ToolEvidenceResult(
            tool_name="search_repository_documents",
            repository=repository,
            items=items,
            message=None if items else "No indexed repository documents matched.",
        )

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolEvidenceResult:
        repository = str(arguments.get("repository", ""))
        if name == "search_issues":
            return await self.search_issues(
                repository, str(arguments["query"]), int(arguments.get("limit", 5))
            )
        if name == "get_issue":
            return await self.get_issue(repository, int(arguments["number"]))
        if name == "search_pull_requests":
            return await self.search_pull_requests(
                repository, str(arguments["query"]), int(arguments.get("limit", 5))
            )
        if name == "get_pull_request":
            return await self.get_pull_request(repository, int(arguments["number"]))
        if name == "get_release":
            return await self.get_release(repository, str(arguments["tag"]))
        if name == "search_repository_documents":
            return await self.search_repository_documents(
                repository, str(arguments["query"]), int(arguments.get("limit", 5))
            )
        raise ValueError(f"Unknown MCP tool: {name}")

    async def _search(
        self, repository: str, query: str, kind: str, limit: int
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("search query must not be blank")
        if not 1 <= limit <= 20:
            raise ValueError("search limit must be between 1 and 20")
        payload = await self.client.get_json(
            "/search/issues",
            {"q": f"repo:{repository} is:{kind} {query}", "per_page": limit},
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        return [item for item in items[:limit] if isinstance(item, dict)]

    @staticmethod
    def _documents(tool_name: str, repository: str, documents: list[Any]) -> ToolEvidenceResult:
        items = [ToolEvidence.from_document(document) for document in documents]
        return ToolEvidenceResult(
            tool_name=tool_name,
            repository=repository,
            items=items,
            message=None if items else "GitHub returned no matching evidence.",
        )

    @staticmethod
    def _positive_number(number: int) -> None:
        if number < 1:
            raise ValueError("issue or pull request number must be positive")
