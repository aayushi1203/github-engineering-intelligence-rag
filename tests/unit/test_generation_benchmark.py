from pathlib import Path

import pytest

from github_intelligence.cli import build_parser
from github_intelligence.generation import (
    AnswerSource,
    Citation,
    GenerationDiagnostic,
    GenerationEvaluator,
    GenerationMetrics,
    GroundedAnswer,
    load_generation_dataset,
)
from github_intelligence.models import DocumentType


def source(source_id: str = "S1") -> AnswerSource:
    return AnswerSource(
        source_id=source_id,
        chunk_id="chunk-1",
        source_type=DocumentType.PULL_REQUEST,
        title="Fix regression",
        source_url="https://github.com/acme/repo/pull/1",
        repository="acme/repo",
        text="The timeout regression was fixed.",
    )


def answer() -> GroundedAnswer:
    evidence = source()
    return GroundedAnswer(
        answer="The timeout regression was fixed [S1].",
        citations=[Citation(source_id="S1", source_url=evidence.source_url)],
        sources=[evidence],
        retrieved_sources=[evidence],
    )


class FakeService:
    def ask(self, question: str, *, repository: str) -> GroundedAnswer:
        if question == "fail":
            raise RuntimeError("provider unavailable")
        return answer()


def write_dataset(path: Path) -> None:
    path.write_text(
        '{"question":"first","repository":"acme/repo"}\n'
        '{"question":"second","repository":"other/repo"}\n',
        encoding="utf-8",
    )


def test_generation_dataset_loading(tmp_path: Path) -> None:
    path = tmp_path / "generation.jsonl"
    write_dataset(path)
    examples = load_generation_dataset(path)
    assert [example.question for example in examples] == ["first", "second"]


def test_generation_dataset_repository_filtering(tmp_path: Path) -> None:
    path = tmp_path / "generation.jsonl"
    write_dataset(path)
    examples = load_generation_dataset(path, repository="other/repo")
    assert len(examples) == 1
    assert examples[0].repository == "other/repo"
    with pytest.raises(ValueError, match="no examples for repository missing/repo"):
        load_generation_dataset(path, repository="missing/repo")


def test_generation_aggregation() -> None:
    diagnostics = [
        GenerationDiagnostic(
            question="one",
            repository="acme/repo",
            metrics=GenerationMetrics(
                citation_validity=1,
                citation_coverage=0.5,
                groundedness=0.25,
                claim_count=4,
                cited_claim_count=2,
            ),
        ),
        GenerationDiagnostic(
            question="two",
            repository="acme/repo",
            metrics=GenerationMetrics(
                citation_validity=0.5,
                citation_coverage=1,
                groundedness=0.75,
                claim_count=2,
                cited_claim_count=2,
            ),
        ),
        GenerationDiagnostic(
            question="failed", repository="acme/repo", error_type="RuntimeError", error="bad"
        ),
    ]
    aggregate = GenerationEvaluator.aggregate(diagnostics)
    assert aggregate.examples_evaluated == 3
    assert aggregate.successful_generations == 2
    assert aggregate.failed_generations == 1
    assert aggregate.mean_citation_validity == 0.75
    assert aggregate.mean_citation_coverage == 0.75
    assert aggregate.mean_groundedness == 0.5
    assert aggregate.total_claim_count == 6
    assert aggregate.total_cited_claim_count == 4


def test_failed_generation_is_recorded_and_batch_continues(tmp_path: Path) -> None:
    examples_path = tmp_path / "generation.jsonl"
    examples_path.write_text(
        '{"question":"fail","repository":"acme/repo"}\n'
        '{"question":"works","repository":"acme/repo"}\n',
        encoding="utf-8",
    )
    report = GenerationEvaluator(FakeService()).evaluate(
        load_generation_dataset(examples_path), dataset_path=examples_path, top_k=8
    )
    assert report.aggregate.successful_generations == 1
    assert report.aggregate.failed_generations == 1
    assert report.diagnostics[0].error_type == "RuntimeError"
    assert report.diagnostics[0].metrics is None
    assert report.diagnostics[1].answer == answer().answer


def test_generation_cli_parser_wiring() -> None:
    args = build_parser().parse_args(
        [
            "evaluate-generation",
            "--dataset",
            "data/eval/generation.jsonl",
            "--repo",
            "encode/httpx",
            "--top-k",
            "5",
        ]
    )
    assert args.command == "evaluate-generation"
    assert args.dataset == Path("data/eval/generation.jsonl")
    assert args.repo == "encode/httpx"
    assert args.top_k == 5
    assert args.output == Path("output/generation-evaluation.json")


@pytest.mark.parametrize(
    "line",
    [
        '{"question":"","repository":"acme/repo"}',
        '{"question":"valid","repository":"bad"}',
        '{"question":"valid","repository":"acme/repo","extra":true}',
    ],
)
def test_invalid_generation_dataset_reports_line(tmp_path: Path, line: str) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_generation_dataset(path)
