from pathlib import Path
from statistics import fmean

from pydantic import ValidationError

from github_intelligence.config import Settings
from github_intelligence.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from github_intelligence.evaluation.models import (
    EvaluationConfig,
    EvaluationExample,
    EvaluationReport,
    ModeMetrics,
    QueryDiagnostic,
    RetrievedDiagnostic,
)
from github_intelligence.models import RetrievalResult
from github_intelligence.retrieval import RetrievalMode, RetrievalPipeline


def load_evaluation_dataset(path: Path) -> list[EvaluationExample]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    examples: list[EvaluationExample] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                examples.append(EvaluationExample.model_validate_json(line))
            except ValidationError as error:
                raise ValueError(
                    f"Invalid evaluation entry at line {line_number}: {error}"
                ) from error
    if not examples:
        raise ValueError("Evaluation dataset must contain at least one example")
    return examples


class RetrievalEvaluator:
    def __init__(self, pipeline: RetrievalPipeline, settings: Settings) -> None:
        self.pipeline = pipeline
        self.settings = settings

    def evaluate(
        self,
        examples: list[EvaluationExample],
        *,
        dataset_path: Path,
        modes: tuple[RetrievalMode, ...] = tuple(RetrievalMode),
        retrieval_depth: int = 10,
    ) -> EvaluationReport:
        if retrieval_depth < 10:
            raise ValueError("retrieval_depth must be at least 10 for Recall@10 and NDCG@10")
        diagnostics: list[QueryDiagnostic] = []
        metrics: dict[RetrievalMode, ModeMetrics] = {}
        for mode in modes:
            mode_diagnostics = [
                self._evaluate_query(example, mode, retrieval_depth) for example in examples
            ]
            diagnostics.extend(mode_diagnostics)
            metrics[mode] = self._aggregate(mode_diagnostics)
        return EvaluationReport(
            dataset_path=str(dataset_path),
            repositories=sorted({example.repository for example in examples}),
            dataset_size=len(examples),
            config=EvaluationConfig(
                embedding_model=self.settings.embedding_model,
                reranker_model=self.settings.reranker_model,
                rrf_k=self.settings.rrf_k,
                candidates_per_backend=self.settings.retrieval_candidates_per_backend,
                rerank_candidate_pool=self.settings.rerank_candidate_pool,
                retrieval_depth=retrieval_depth,
            ),
            metrics=metrics,
            diagnostics=diagnostics,
        )

    def _evaluate_query(
        self, example: EvaluationExample, mode: RetrievalMode, retrieval_depth: int
    ) -> QueryDiagnostic:
        results = self.pipeline.search(
            example.query,
            mode=mode,
            top_k=retrieval_depth,
            repository=example.repository,
        )
        relevant = set(example.relevant_source_ids)
        retrieved = self._diagnostics(results, relevant)
        retrieved_ids = [item.source_id for item in retrieved]
        first_rank = next(
            (
                rank
                for rank, source_id in enumerate(retrieved_ids, start=1)
                if source_id in relevant
            ),
            None,
        )
        recall_5 = recall_at_k(retrieved_ids, relevant, 5)
        recall_10 = recall_at_k(retrieved_ids, relevant, 10)
        rr = reciprocal_rank(retrieved_ids, relevant)
        ndcg_10 = ndcg_at_k(retrieved_ids, relevant, 10)
        return QueryDiagnostic(
            mode=mode,
            repository=example.repository,
            query=example.query,
            relevant_source_ids=example.relevant_source_ids,
            retrieved=retrieved,
            first_relevant_rank=first_rank,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            reciprocal_rank=rr,
            ndcg_at_10=ndcg_10,
            succeeded_at_5=recall_5 > 0,
            succeeded_at_10=recall_10 > 0,
        )

    @staticmethod
    def _diagnostics(
        results: list[RetrievalResult], relevant: set[str]
    ) -> list[RetrievedDiagnostic]:
        diagnostics: list[RetrievedDiagnostic] = []
        seen_sources: set[str] = set()
        for result in results:
            source_id = _source_id(result, relevant)
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            diagnostics.append(
                RetrievedDiagnostic(
                    source_id=source_id,
                    chunk_id=result.chunk_id,
                    rank=len(diagnostics) + 1,
                    score=result.score,
                    bm25_score=result.bm25_score,
                    vector_score=result.vector_score,
                    rrf_score=result.rrf_score,
                    reranker_score=result.reranker_score,
                )
            )
        return diagnostics

    @staticmethod
    def _aggregate(diagnostics: list[QueryDiagnostic]) -> ModeMetrics:
        if not diagnostics:
            return ModeMetrics(recall_at_5=0, recall_at_10=0, mrr=0, ndcg_at_10=0, query_count=0)
        return ModeMetrics(
            recall_at_5=fmean(item.recall_at_5 for item in diagnostics),
            recall_at_10=fmean(item.recall_at_10 for item in diagnostics),
            mrr=fmean(item.reciprocal_rank for item in diagnostics),
            ndcg_at_10=fmean(item.ndcg_at_10 for item in diagnostics),
            query_count=len(diagnostics),
        )


def _source_id(result: RetrievalResult, relevant: set[str]) -> str:
    aliases = [
        result.chunk_id,
        str(result.metadata.get("github_id", "")),
        str(result.metadata.get("document_id", "")),
    ]
    for alias in aliases:
        if alias and alias in relevant:
            return alias
    for alias in aliases[1:]:
        if alias:
            return alias
    return result.chunk_id
