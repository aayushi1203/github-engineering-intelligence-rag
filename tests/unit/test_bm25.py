from pathlib import Path

from github_intelligence.models import DocumentChunk, DocumentType
from github_intelligence.retrieval.bm25 import BM25Retriever


def chunk(identifier: str, text: str, repository: str = "encode/httpx") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        document_id=identifier,
        chunk_index=0,
        text=text,
        title=text,
        source_url=f"https://github.com/{repository}/issues/1",
        source_type=DocumentType.ISSUE,
        repository=repository,
        metadata={"evidence": identifier},
    )


def test_bm25_ranks_lexical_match_and_persists_metadata(tmp_path: Path) -> None:
    index_path = tmp_path / "chunks.jsonl"
    retriever = BM25Retriever(index_path)
    retriever.index(
        [
            chunk("auth", "authentication timeout regression"),
            chunk("release", "release packaging wheels"),
            chunk("docs", "documentation redirects proxies"),
        ]
    )
    reloaded = BM25Retriever(index_path)
    results = reloaded.search("authentication regression", top_k=2, repository="encode/httpx")
    assert results[0].chunk_id == "auth"
    assert results[0].metadata["evidence"] == "auth"
    assert str(results[0].source_url).endswith("/issues/1")


def test_bm25_handles_empty_index_query_and_repository_filter(tmp_path: Path) -> None:
    retriever = BM25Retriever(tmp_path / "missing.jsonl")
    assert retriever.search("authentication") == []
    retriever.index([chunk("auth", "authentication timeout")])
    assert retriever.search("") == []
    assert retriever.search("unrelated-vocabulary") == []
    assert retriever.search("authentication")[0].chunk_id == "auth"
    assert retriever.search("authentication", repository="other/repo") == []
