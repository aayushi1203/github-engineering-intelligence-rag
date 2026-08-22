from enum import Enum
from typing import TypeVar

from github_intelligence.models import RetrievalResult
from github_intelligence.retrieval.base import Retriever
from github_intelligence.retrieval.fusion import reciprocal_rank_fusion
from github_intelligence.retrieval.reranker import Reranker

T = TypeVar("T")


class RetrievalMode(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RetrievalPipeline:
    """Orchestrates single-backend, RRF hybrid, and reranked retrieval modes."""

    def __init__(
        self,
        *,
        bm25: Retriever | None = None,
        vector: Retriever | None = None,
        reranker: Reranker | None = None,
        candidates_per_backend: int = 25,
        rerank_candidate_pool: int = 40,
        rrf_k: int = 60,
    ) -> None:
        if candidates_per_backend < 1 or rerank_candidate_pool < 1 or rrf_k < 1:
            raise ValueError("candidate pool sizes and rrf_k must be positive")
        self.bm25 = bm25
        self.vector = vector
        self.reranker = reranker
        self.candidates_per_backend = candidates_per_backend
        self.rerank_candidate_pool = rerank_candidate_pool
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID_RERANK,
        top_k: int = 10,
        repository: str | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip() or top_k <= 0:
            return []
        if mode == RetrievalMode.BM25:
            results = self._require(self.bm25, "BM25").search(
                query, top_k=top_k, repository=repository
            )
            return self._single_backend(results, "bm25")
        if mode == RetrievalMode.VECTOR:
            results = self._require(self.vector, "vector").search(
                query, top_k=top_k, repository=repository
            )
            return self._single_backend(results, "vector")

        bm25_results = self._require(self.bm25, "BM25").search(
            query, top_k=self.candidates_per_backend, repository=repository
        )
        vector_results = self._require(self.vector, "vector").search(
            query, top_k=self.candidates_per_backend, repository=repository
        )
        fused = reciprocal_rank_fusion(bm25_results, vector_results, rrf_k=self.rrf_k)
        if mode == RetrievalMode.HYBRID:
            return self._finalize(fused[:top_k])
        reranker = self._require(self.reranker, "reranker")
        return reranker.rerank(
            query, fused[: self.rerank_candidate_pool], top_k=top_k
        )

    @staticmethod
    def _require(component: T | None, name: str) -> T:
        if component is None:
            raise RuntimeError(f"{name} component is required for this retrieval mode")
        return component

    @staticmethod
    def _single_backend(
        results: list[RetrievalResult], backend: str
    ) -> list[RetrievalResult]:
        enriched = []
        for rank, result in enumerate(results, start=1):
            fields = {f"{backend}_rank": rank, f"{backend}_score": result.score}
            enriched.append(result.model_copy(update=fields | {"final_rank": rank}))
        return enriched

    @staticmethod
    def _finalize(results: list[RetrievalResult]) -> list[RetrievalResult]:
        return [
            result.model_copy(update={"final_rank": rank})
            for rank, result in enumerate(results, start=1)
        ]
