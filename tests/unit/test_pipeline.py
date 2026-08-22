from collections.abc import Sequence

from github_intelligence.models import DocumentType, RetrievalResult
from github_intelligence.retrieval.pipeline import RetrievalMode, RetrievalPipeline


def result(identifier: str, score: float, repository: str = "encode/httpx") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=identifier,
        text=f"Evidence about {identifier}",
        title=f"Title {identifier}",
        source_url=f"https://github.com/{repository}/issues/{identifier}",
        source_type=DocumentType.ISSUE,
        score=score,
        repository=repository,
        metadata={"stable": identifier},
    )


class FakeRetriever:
    def __init__(self, results: Sequence[RetrievalResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int, str | None]] = []

    def index(self, chunks: Sequence[object]) -> None:
        pass

    def search(
        self, query: str, *, top_k: int = 10, repository: str | None = None
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k, repository))
        filtered = [
            item for item in self.results if not repository or item.repository == repository
        ]
        return filtered[:top_k]


class FakeReranker:
    def rerank(
        self, query: str, results: Sequence[RetrievalResult], *, top_k: int
    ) -> list[RetrievalResult]:
        scored = [
            item.model_copy(
                update={
                    "score": 10.0 if item.chunk_id == "semantic" else 1.0,
                    "reranker_score": 10.0 if item.chunk_id == "semantic" else 1.0,
                }
            )
            for item in results
        ]
        scored.sort(key=lambda item: (-float(item.reranker_score or 0), item.chunk_id))
        return [
            item.model_copy(update={"final_rank": rank})
            for rank, item in enumerate(scored[:top_k], start=1)
        ]


def test_hybrid_pipeline_fuses_deduplicates_and_filters_repository() -> None:
    bm25 = FakeRetriever([result("shared", 8.0), result("lexical", 7.0)])
    vector = FakeRetriever(
        [result("shared", 0.9), result("semantic", 0.8), result("foreign", 1.0, "other/repo")]
    )
    pipeline = RetrievalPipeline(bm25=bm25, vector=vector, candidates_per_backend=5)
    results = pipeline.search(
        "auth regression", mode=RetrievalMode.HYBRID, top_k=3, repository="encode/httpx"
    )
    assert [item.chunk_id for item in results] == ["shared", "lexical", "semantic"]
    assert len({item.chunk_id for item in results}) == 3
    assert all(item.repository == "encode/httpx" for item in results)
    assert results[0].metadata["stable"] == "shared"
    assert [item.final_rank for item in results] == [1, 2, 3]
    assert bm25.calls[0][2] == vector.calls[0][2] == "encode/httpx"


def test_hybrid_rerank_changes_order_and_retains_rrf_scores() -> None:
    pipeline = RetrievalPipeline(
        bm25=FakeRetriever([result("shared", 8.0), result("lexical", 7.0)]),
        vector=FakeRetriever([result("shared", 0.9), result("semantic", 0.8)]),
        reranker=FakeReranker(),
        rerank_candidate_pool=3,
    )
    results = pipeline.search("semantic authentication", mode=RetrievalMode.HYBRID_RERANK, top_k=2)
    assert [item.chunk_id for item in results] == ["semantic", "lexical"]
    assert results[0].reranker_score == 10.0
    assert results[0].rrf_score is not None
    assert results[0].vector_rank == 2
    assert [item.final_rank for item in results] == [1, 2]


def test_single_modes_expose_backend_rank_and_empty_is_stable() -> None:
    bm25 = FakeRetriever([result("lexical", 4.0)])
    vector = FakeRetriever([])
    pipeline = RetrievalPipeline(bm25=bm25, vector=vector)
    lexical = pipeline.search("auth", mode=RetrievalMode.BM25, top_k=1)
    assert lexical[0].bm25_rank == lexical[0].final_rank == 1
    assert lexical[0].bm25_score == 4.0
    assert pipeline.search("auth", mode=RetrievalMode.VECTOR) == []
    assert pipeline.search("", mode=RetrievalMode.HYBRID) == []
