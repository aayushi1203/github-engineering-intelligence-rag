from collections.abc import Sequence
from typing import Protocol

from github_intelligence.models import DocumentChunk, RetrievalResult


class Retriever(Protocol):
    """Common contract implemented by lexical and semantic retrieval."""

    def index(self, chunks: Sequence[DocumentChunk]) -> None: ...

    def search(
        self, query: str, *, top_k: int = 10, repository: str | None = None
    ) -> list[RetrievalResult]: ...
