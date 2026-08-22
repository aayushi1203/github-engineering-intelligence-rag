from pathlib import Path

import pytest

from github_intelligence.config import Settings
from github_intelligence.evaluation import EvaluationExample, RetrievalEvaluator
from github_intelligence.evaluation.evaluator import load_evaluation_dataset
from github_intelligence.models import DocumentType, RetrievalResult
from github_intelligence.retrieval import RetrievalMode


def result(identifier: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{identifier}",
        text=f"Evidence {identifier}",
        title=f"Issue {identifier}",
        source_url=f"https://github.com/encode/httpx/issues/{identifier}",
        source_type=DocumentType.ISSUE,
        score=1.0,
        repository="encode/httpx",
        metadata={"github_id": identifier},
    )


class FakePipeline:
    def search(self, query: str, **kwargs):  # type: ignore[no-untyped-def]
        return [result("a"), result("x")] if query == "found" else [result("x")]


def test_evaluation_aggregation_and_source_id_matching(tmp_path: Path) -> None:
    examples = [
        EvaluationExample(
            repository="encode/httpx", query="found", relevant_source_ids=["a"]
        ),
        EvaluationExample(
            repository="encode/httpx", query="missing", relevant_source_ids=["b"]
        ),
    ]
    evaluator = RetrievalEvaluator(FakePipeline(), Settings())  # type: ignore[arg-type]
    report = evaluator.evaluate(
        examples,
        dataset_path=tmp_path / "eval.jsonl",
        modes=(RetrievalMode.BM25,),
        retrieval_depth=10,
    )
    metrics = report.metrics[RetrievalMode.BM25]
    assert metrics.query_count == 2
    assert metrics.recall_at_5 == metrics.recall_at_10 == 0.5
    assert metrics.mrr == 0.5
    assert metrics.ndcg_at_10 == 0.5
    assert report.diagnostics[0].retrieved[0].source_id == "a"
    assert report.diagnostics[0].first_relevant_rank == 1
    assert report.diagnostics[1].first_relevant_rank is None


@pytest.mark.parametrize(
    "line",
    [
        '{"repository":"bad","query":"q","relevant_source_ids":["1"]}',
        '{"repository":"o/r","query":" ","relevant_source_ids":["1"]}',
        '{"repository":"o/r","query":"q","relevant_source_ids":[]}',
        '{"repository":"o/r","query":"q","relevant_source_ids":["1","1"]}',
        '{"repository":"o/r","query":"q","relevant_source_ids":["1"],"unknown":true}',
    ],
)
def test_invalid_dataset_entries_report_line_number(tmp_path: Path, line: str) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_evaluation_dataset(path)


def test_starter_dataset_contains_verified_examples() -> None:
    path = Path(__file__).parents[2] / "data/eval/retrieval.jsonl"
    examples = load_evaluation_dataset(path)
    assert len(examples) == 20
    assert {example.repository for example in examples} == {
        "fastapi/fastapi",
        "pydantic/pydantic",
    }
    assert all(example.notes and "https://github.com/" in example.notes for example in examples)
