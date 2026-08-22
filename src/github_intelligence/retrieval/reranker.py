from collections.abc import Sequence
from typing import Any, Protocol

from github_intelligence.models import RetrievalResult


class Reranker(Protocol):
    def rerank(
        self, query: str, results: Sequence[RetrievalResult], *, top_k: int
    ) -> list[RetrievalResult]: ...


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder reranker."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self, query: str, results: Sequence[RetrievalResult], *, top_k: int
    ) -> list[RetrievalResult]:
        if not query.strip() or not results or top_k <= 0:
            return []
        pairs = [(query, result.text) for result in results]
        raw_scores = self._load().predict(
            pairs, batch_size=self.batch_size, show_progress_bar=len(pairs) > 100
        )
        scored = [
            result.model_copy(
                update={"score": float(score), "reranker_score": float(score)}
            )
            for result, score in zip(results, raw_scores, strict=True)
        ]
        scored.sort(
            key=lambda result: (
                -float(result.reranker_score or 0.0),
                -(result.rrf_score or 0.0),
                result.chunk_id,
            )
        )
        return [
            result.model_copy(update={"final_rank": rank})
            for rank, result in enumerate(scored[:top_k], start=1)
        ]
