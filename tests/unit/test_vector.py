from collections.abc import Sequence

from github_intelligence.models import DocumentChunk, DocumentType
from github_intelligence.retrieval.vector import QdrantVectorRetriever, VectorMatch


class FakeEncoder:
    @property
    def dimension(self) -> int:
        return 3

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float("auth" in lower or "authentication" in lower),
                    float("documentation" in lower),
                    float("release" in lower),
                ]
            )
        return vectors


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: list[tuple[DocumentChunk, Sequence[float]]] | None = None

    def exists(self) -> bool:
        return self.records is not None

    def replace(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        dimension: int,
    ) -> None:
        assert dimension == 3
        self.records = list(zip(chunks, vectors, strict=True))

    def search(
        self, vector: Sequence[float], *, top_k: int, repository: str | None
    ) -> list[VectorMatch]:
        records = self.records or []
        matches = [
            VectorMatch(chunk, sum(left * right for left, right in zip(vector, stored, strict=True)))
            for chunk, stored in records
            if repository is None or chunk.repository == repository
        ]
        return sorted(matches, key=lambda match: match.score, reverse=True)[:top_k]


def chunk(identifier: str, text: str, repository: str = "encode/httpx") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        document_id=identifier,
        chunk_index=0,
        text=text,
        title=f"Title {identifier}",
        source_url=f"https://github.com/{repository}/issues/{identifier}",
        source_type=DocumentType.ISSUE,
        repository=repository,
        metadata={"parent_number": 7},
    )


def test_qdrant_indexes_searches_and_preserves_metadata() -> None:
    retriever = QdrantVectorRetriever(FakeVectorStore(), FakeEncoder())
    auth_id = "00000000-0000-0000-0000-000000000001"
    release_id = "00000000-0000-0000-0000-000000000002"
    retriever.index(
        [chunk(auth_id, "authentication regression"), chunk(release_id, "release notes")]
    )
    results = retriever.search("auth timeout", top_k=1, repository="encode/httpx")
    assert results[0].chunk_id == auth_id
    assert results[0].source_type == DocumentType.ISSUE
    assert results[0].metadata["parent_number"] == 7
    assert str(results[0].source_url).endswith(f"/issues/{auth_id}")


def test_qdrant_handles_missing_collection_empty_query_and_no_results() -> None:
    retriever = QdrantVectorRetriever(FakeVectorStore(), FakeEncoder())
    assert retriever.search("auth") == []
    retriever.index(
        [chunk("00000000-0000-0000-0000-000000000001", "authentication", "encode/httpx")]
    )
    assert retriever.search("") == []
    assert retriever.search("auth", repository="other/repo") == []
