from github_intelligence.models import DocumentType, RetrievalResult
from github_intelligence.retrieval.reranker import CrossEncoderReranker


def result(identifier: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=identifier,
        text=f"Evidence {identifier}",
        title=f"Title {identifier}",
        source_url=f"https://github.com/encode/httpx/issues/{identifier}",
        source_type=DocumentType.ISSUE,
        score=score,
        repository="encode/httpx",
    )


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):  # type: ignore[no-untyped-def]
        return [0.1, 0.9]


def test_cross_encoder_reranker_orders_scores_deterministically() -> None:
    reranker = CrossEncoderReranker("fake")
    reranker._model = FakeCrossEncoder()
    results = reranker.rerank("auth", [result("a", 0.2), result("b", 0.1)], top_k=2)
    assert [item.chunk_id for item in results] == ["b", "a"]
    assert [item.reranker_score for item in results] == [0.9, 0.1]
    assert [item.final_rank for item in results] == [1, 2]


def test_cross_encoder_reranker_handles_empty_inputs() -> None:
    reranker = CrossEncoderReranker("fake")
    assert reranker.rerank("auth", [], top_k=5) == []
    assert reranker.rerank("", [result("a", 1.0)], top_k=5) == []
