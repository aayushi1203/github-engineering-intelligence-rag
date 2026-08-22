from collections.abc import Sequence

from github_intelligence.models import RetrievalResult


def reciprocal_rank_fusion(
    bm25_results: Sequence[RetrievalResult],
    vector_results: Sequence[RetrievalResult],
    *,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse two ranked lists by stable chunk ID using one-based RRF ranks."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    entries: dict[str, RetrievalResult] = {}
    contributions: dict[str, float] = {}
    best_ranks: dict[str, int] = {}

    for backend, results in (("bm25", bm25_results), ("vector", vector_results)):
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            existing = entries.get(result.chunk_id, result)
            update: dict[str, int | float] = {}
            if backend == "bm25":
                update = {"bm25_rank": rank, "bm25_score": result.score}
            else:
                update = {"vector_rank": rank, "vector_score": result.score}
            entries[result.chunk_id] = existing.model_copy(update=update)
            contributions[result.chunk_id] = contributions.get(result.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
            best_ranks[result.chunk_id] = min(best_ranks.get(result.chunk_id, rank), rank)

    fused = [
        result.model_copy(
            update={"score": contributions[chunk_id], "rrf_score": contributions[chunk_id]}
        )
        for chunk_id, result in entries.items()
    ]
    fused.sort(
        key=lambda result: (
            -contributions[result.chunk_id],
            best_ranks[result.chunk_id],
            result.chunk_id,
        )
    )
    return fused
