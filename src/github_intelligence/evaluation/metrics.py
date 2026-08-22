import math
from collections.abc import Iterable, Sequence


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant must contain at least one source ID")
    if k < 1:
        raise ValueError("k must be at least 1")
    hits = relevant.intersection(_unique(retrieved)[:k])
    return len(hits) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        raise ValueError("relevant must contain at least one source ID")
    for rank, source_id in enumerate(_unique(retrieved), start=1):
        if source_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rankings: Sequence[Sequence[str]], labels: Sequence[set[str]]) -> float:
    if len(rankings) != len(labels):
        raise ValueError("rankings and labels must have equal length")
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(ranking, relevant)
        for ranking, relevant in zip(rankings, labels)  # noqa: B905
    ) / len(rankings)


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        raise ValueError("relevant must contain at least one source ID")
    if k < 1:
        raise ValueError("k must be at least 1")
    ranking = _unique(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source_id in enumerate(ranking, start=1)
        if source_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg
