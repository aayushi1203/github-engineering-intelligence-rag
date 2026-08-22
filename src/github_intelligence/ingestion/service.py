import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog

from github_intelligence.github import GitHubClient
from github_intelligence.ingestion.normalizers import (
    normalize_comment,
    normalize_commit,
    normalize_documentation,
    normalize_issue,
    normalize_pull_request,
    normalize_release,
)
from github_intelligence.models import GitHubDocument

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionOptions:
    include_comments: bool = True
    include_documentation: bool = True


class RepositoryIngestor:
    def __init__(self, client: GitHubClient, doc_max_bytes: int = 1_000_000) -> None:
        self.client = client
        self.doc_max_bytes = doc_max_bytes

    async def ingest(
        self, repository: str, options: IngestionOptions | None = None
    ) -> AsyncIterator[GitHubDocument]:
        if repository.count("/") != 1:
            raise ValueError("repository must use the owner/name format")
        options = options or IngestionOptions()
        logger.info("ingestion_started", repository=repository)
        async for document in self._issues(repository, options.include_comments):
            yield document
        async for document in self._pull_requests(repository, options.include_comments):
            yield document
        async for item in self.client.paginate(f"/repos/{repository}/commits"):
            yield normalize_commit(repository, item)
        async for item in self.client.paginate(f"/repos/{repository}/releases"):
            yield normalize_release(repository, item)
        if options.include_documentation:
            async for document in self._documentation(repository):
                yield document
        logger.info("ingestion_completed", repository=repository)

    async def _issues(self, repo: str, comments: bool) -> AsyncIterator[GitHubDocument]:
        async for item in self.client.paginate(f"/repos/{repo}/issues", {"state": "all"}):
            if "pull_request" in item:
                continue
            yield normalize_issue(repo, item)
            if comments and item.get("comments", 0):
                async for comment in self.client.paginate(
                    f"/repos/{repo}/issues/{item['number']}/comments"
                ):
                    yield normalize_comment(repo, comment, item["number"], pull_request=False)

    async def _pull_requests(self, repo: str, comments: bool) -> AsyncIterator[GitHubDocument]:
        async for item in self.client.paginate(f"/repos/{repo}/pulls", {"state": "all"}):
            yield normalize_pull_request(repo, item)
            if comments:
                # Issue comments are the PR conversation; review comments are inline code comments.
                async for comment in self.client.paginate(
                    f"/repos/{repo}/issues/{item['number']}/comments"
                ):
                    yield normalize_comment(repo, comment, item["number"], pull_request=True)
                async for comment in self.client.paginate(
                    f"/repos/{repo}/pulls/{item['number']}/comments"
                ):
                    yield normalize_comment(
                        repo, comment, item["number"], pull_request=True, review=True
                    )

    async def _documentation(self, repo: str) -> AsyncIterator[GitHubDocument]:
        repository = await self.client.get_json(f"/repos/{repo}")
        branch = repository["default_branch"]
        tree = await self.client.get_json(f"/repos/{repo}/git/trees/{branch}", {"recursive": "1"})
        if tree.get("truncated"):
            logger.warning("documentation_tree_truncated", repository=repo)
        for entry in tree.get("tree", []):
            if not self._is_documentation(entry):
                continue
            if int(entry.get("size") or 0) > self.doc_max_bytes:
                logger.info("documentation_skipped_too_large", path=entry.get("path"))
                continue
            path = entry["path"]
            payload = await self.client.get_json(f"/repos/{repo}/contents/{path}", {"ref": branch})
            content = self._decode_content(payload)
            if not content.strip():
                continue
            html_url = payload.get("html_url") or f"https://github.com/{repo}/blob/{branch}/{path}"
            yield normalize_documentation(repo, path, entry["sha"], html_url, content)

    @staticmethod
    def _is_documentation(entry: dict[str, Any]) -> bool:
        if entry.get("type") != "blob":
            return False
        path = str(entry.get("path", ""))
        lower = path.lower()
        name = lower.rsplit("/", 1)[-1]
        return (
            name.startswith("readme")
            or lower.startswith(("docs/", "doc/", "documentation/"))
            or ("/docs/" in lower and lower.endswith((".md", ".mdx", ".rst", ".txt")))
        ) and lower.endswith((".md", ".mdx", ".rst", ".txt", "readme"))

    @staticmethod
    def _decode_content(payload: dict[str, Any]) -> str:
        encoded = payload.get("content")
        if payload.get("encoding") != "base64" or not isinstance(encoded, str):
            return ""
        return base64.b64decode(encoded).decode("utf-8", errors="replace")

