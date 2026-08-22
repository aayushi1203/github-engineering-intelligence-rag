import pytest

from github_intelligence.models import DocumentType, RetrievalResult
from github_intelligence.retrieval.fusion import reciprocal_rank_fusion


def result(identifier: str, score: float, metadata: dict[str, str] | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=identifier,
        text=f"Evidence {identifier}",
        title=f"Title {identifier}",
        source_url=f"https://github.com/encode/httpx/issues/{identifier}",
        source_type=DocumentType.ISSUE,
        score=score,
        repository="encode/httpx",
        metadata=metadata or {},
    )


def test_rrf_rewards_agreement_and_preserves_backend_diagnostics() -> None:
    fused = reciprocal_rank_fusion(
        [result("a", 9.0), result("b", 7.0, {"parent": "issue-1"})],
        [result("b", 0.95), result("c", 0.8)],
        rrf_k=60,
    )
    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].bm25_rank == 2
    assert fused[0].bm25_score == 7.0
    assert fused[0].vector_rank == 1
    assert fused[0].vector_score == 0.95
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].metadata["parent"] == "issue-1"
    assert str(fused[0].source_url).endswith("/issues/b")


def test_rrf_deduplicates_and_breaks_ties_by_chunk_id() -> None:
    duplicate = result("b", 6.0)
    fused = reciprocal_rank_fusion(
        [result("b", 7.0), duplicate, result("a", 5.0)], [], rrf_k=60
    )
    assert [item.chunk_id for item in fused] == ["b", "a"]
    assert fused[0].bm25_score == 7.0


def test_rrf_empty_results_are_empty() -> None:
    assert reciprocal_rank_fusion([], []) == []
