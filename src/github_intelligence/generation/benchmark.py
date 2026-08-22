from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from github_intelligence.generation.evaluation import evaluate_grounded_answer
from github_intelligence.generation.models import AnswerSource, GenerationMetrics, GroundedAnswer


class GenerationEvaluationExample(BaseModel):
    """One question for online generation evaluation."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class GenerationAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    examples_evaluated: int = Field(ge=0)
    successful_generations: int = Field(ge=0)
    failed_generations: int = Field(ge=0)
    mean_citation_validity: float | None = Field(default=None, ge=0, le=1)
    mean_citation_coverage: float | None = Field(default=None, ge=0, le=1)
    mean_groundedness: float | None = Field(default=None, ge=0, le=1)
    total_claim_count: int = Field(ge=0)
    total_cited_claim_count: int = Field(ge=0)


class GenerationDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    repository: str
    answer: str | None = None
    sources: list[AnswerSource] = Field(default_factory=list)
    limitations: list[str] | None = None
    metrics: GenerationMetrics | None = None
    error_type: str | None = None
    error: str | None = None


class GenerationEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_path: str
    repositories: list[str]
    top_k: int = Field(ge=1)
    aggregate: GenerationAggregate
    diagnostics: list[GenerationDiagnostic]


class AnswerGenerator(Protocol):
    def ask(self, question: str, *, repository: str) -> GroundedAnswer: ...


def load_generation_dataset(
    path: Path, *, repository: str | None = None
) -> list[GenerationEvaluationExample]:
    if not path.exists():
        raise FileNotFoundError(f"Generation evaluation dataset does not exist: {path}")
    examples: list[GenerationEvaluationExample] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                example = GenerationEvaluationExample.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"Invalid generation evaluation entry at line {line_number}: {error}"
                ) from error
            if repository is None or example.repository == repository:
                examples.append(example)
    if not examples:
        detail = f" for repository {repository}" if repository else ""
        raise ValueError(f"Generation evaluation dataset has no examples{detail}")
    return examples


class GenerationEvaluator:
    def __init__(
        self,
        service: AnswerGenerator,
        metric_function: Callable[[GroundedAnswer], GenerationMetrics] = evaluate_grounded_answer,
    ) -> None:
        self.service = service
        self.metric_function = metric_function

    def evaluate(
        self,
        examples: list[GenerationEvaluationExample],
        *,
        dataset_path: Path,
        top_k: int,
    ) -> GenerationEvaluationReport:
        diagnostics = [self._evaluate_example(example) for example in examples]
        return GenerationEvaluationReport(
            dataset_path=str(dataset_path),
            repositories=sorted({example.repository for example in examples}),
            top_k=top_k,
            aggregate=self.aggregate(diagnostics),
            diagnostics=diagnostics,
        )

    def _evaluate_example(self, example: GenerationEvaluationExample) -> GenerationDiagnostic:
        try:
            answer = self.service.ask(example.question, repository=example.repository)
            metrics = self.metric_function(answer)
            return GenerationDiagnostic(
                question=example.question,
                repository=example.repository,
                answer=answer.answer,
                sources=answer.sources,
                limitations=answer.limitations,
                metrics=metrics,
            )
        except Exception as error:  # continue the paid batch while preserving each failure
            return GenerationDiagnostic(
                question=example.question,
                repository=example.repository,
                error_type=type(error).__name__,
                error=str(error),
            )

    @staticmethod
    def aggregate(diagnostics: list[GenerationDiagnostic]) -> GenerationAggregate:
        metrics = [item.metrics for item in diagnostics if item.metrics is not None]
        return GenerationAggregate(
            examples_evaluated=len(diagnostics),
            successful_generations=len(metrics),
            failed_generations=len(diagnostics) - len(metrics),
            mean_citation_validity=(
                fmean(item.citation_validity for item in metrics) if metrics else None
            ),
            mean_citation_coverage=(
                fmean(item.citation_coverage for item in metrics) if metrics else None
            ),
            mean_groundedness=(fmean(item.groundedness for item in metrics) if metrics else None),
            total_claim_count=sum(item.claim_count for item in metrics),
            total_cited_claim_count=sum(item.cited_claim_count for item in metrics),
        )
