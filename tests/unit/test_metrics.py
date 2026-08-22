import math

import pytest

from github_intelligence.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_perfect_ranking_with_multiple_relevant_sources() -> None:
    relevant = {"a", "b"}
    ranking = ["a", "b", "c"]
    assert recall_at_k(ranking, relevant, 5) == 1.0
    assert reciprocal_rank(ranking, relevant) == 1.0
    assert ndcg_at_k(ranking, relevant, 10) == 1.0


def test_partially_correct_ranking() -> None:
    relevant = {"a", "b"}
    ranking = ["x", "a", "y"]
    assert recall_at_k(ranking, relevant, 3) == 0.5
    assert reciprocal_rank(ranking, relevant) == 0.5
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, relevant, 10) == pytest.approx(expected)


def test_no_relevant_results_scores_zero() -> None:
    relevant = {"a"}
    ranking = ["x", "y"]
    assert recall_at_k(ranking, relevant, 5) == 0.0
    assert reciprocal_rank(ranking, relevant) == 0.0
    assert ndcg_at_k(ranking, relevant, 10) == 0.0


def test_duplicate_retrieved_source_counts_once() -> None:
    relevant = {"a", "b"}
    ranking = ["a", "a", "b"]
    assert recall_at_k(ranking, relevant, 2) == 1.0
    assert ndcg_at_k(ranking, relevant, 2) == 1.0


def test_mean_reciprocal_rank_aggregates_queries() -> None:
    assert mean_reciprocal_rank([["a"], ["x", "b"]], [{"a"}, {"b"}]) == 0.75
    assert mean_reciprocal_rank([], []) == 0.0


def test_metrics_reject_empty_relevance_labels() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 10)
