import asyncio

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as FastMCPSettings
from mcp.types import ToolAnnotations

from github_intelligence.config import Settings
from github_intelligence.github import GitHubClient
from github_intelligence.mcp.models import ToolEvidenceResult
from github_intelligence.mcp.tools import GitHubEngineeringTools
from github_intelligence.retrieval import RetrievalMode

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
FastMCPSettings.model_rebuild()


def create_mcp_server(tools: GitHubEngineeringTools) -> FastMCP[None]:
    server: FastMCP[None] = FastMCP(
        "GitHub Engineering Intelligence",
        instructions="Read-only tools for investigating public GitHub engineering evidence.",
    )

    @server.tool(annotations=READ_ONLY)
    async def search_issues(repository: str, query: str, limit: int = 5) -> ToolEvidenceResult:
        """Search issues in one owner/name repository."""
        return await tools.search_issues(repository, query, limit)

    @server.tool(annotations=READ_ONLY)
    async def get_issue(repository: str, number: int) -> ToolEvidenceResult:
        """Get an issue and its comments by issue number."""
        return await tools.get_issue(repository, number)

    @server.tool(annotations=READ_ONLY)
    async def search_pull_requests(
        repository: str, query: str, limit: int = 5
    ) -> ToolEvidenceResult:
        """Search pull requests in one owner/name repository."""
        return await tools.search_pull_requests(repository, query, limit)

    @server.tool(annotations=READ_ONLY)
    async def get_pull_request(repository: str, number: int) -> ToolEvidenceResult:
        """Get a pull request, conversation comments, and review comments."""
        return await tools.get_pull_request(repository, number)

    @server.tool(annotations=READ_ONLY)
    async def get_release(repository: str, tag: str) -> ToolEvidenceResult:
        """Get one release by its exact Git tag."""
        return await tools.get_release(repository, tag)

    @server.tool(annotations=READ_ONLY)
    async def search_repository_documents(
        repository: str, query: str, limit: int = 5
    ) -> ToolEvidenceResult:
        """Search the local hybrid-reranked repository corpus."""
        return await tools.search_repository_documents(repository, query, limit)

    return server


def main() -> None:
    from github_intelligence.cli import build_retrieval_pipeline

    settings = Settings()
    client = GitHubClient(settings)
    pipeline = build_retrieval_pipeline(settings, {RetrievalMode.HYBRID_RERANK})
    server = create_mcp_server(GitHubEngineeringTools(client, pipeline))
    try:
        server.run(transport="stdio")
    finally:
        asyncio.run(client.aclose())
