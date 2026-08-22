from typing import Any

import pytest

from github_intelligence.ingestion import IngestionOptions, RepositoryIngestor
from github_intelligence.models import DocumentType


class FakeClient:
    async def paginate(self, url: str, params: Any = None):  # type: ignore[no-untyped-def]
        if url.endswith("/issues"):
            yield {"id": 1, "number": 1, "title": "Bug", "body": "broken",
                   "html_url": "https://github.com/o/r/issues/1", "comments": 0, "labels": []}
        elif url.endswith("/pulls"):
            yield {"id": 2, "number": 2, "title": "Fix", "body": "fixed",
                   "html_url": "https://github.com/o/r/pull/2"}
        elif url.endswith("/commits"):
            yield {"sha": "abc", "html_url": "https://github.com/o/r/commit/abc",
                   "commit": {"message": "fix auth", "author": {"date": "2026-01-01T00:00:00Z"}}}
        elif url.endswith("/releases"):
            yield {"id": 3, "tag_name": "v1", "html_url": "https://github.com/o/r/releases/tag/v1"}


@pytest.mark.asyncio
async def test_ingestion_yields_requested_core_types() -> None:
    ingestor = RepositoryIngestor(FakeClient())  # type: ignore[arg-type]
    documents = [doc async for doc in ingestor.ingest(
        "o/r", IngestionOptions(include_comments=False, include_documentation=False)
    )]
    assert [doc.document_type for doc in documents] == [
        DocumentType.ISSUE, DocumentType.PULL_REQUEST, DocumentType.COMMIT, DocumentType.RELEASE
    ]
