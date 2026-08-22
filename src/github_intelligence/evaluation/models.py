from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from github_intelligence.retrieval.pipeline import RetrievalMode


class EvaluationExample(BaseModel):
    """One source-labeled offline retrieval query."""

    model_config = ConfigDict(extra="forbid")

    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    query: str = Field(min_length=1)
    relevant_source_ids: list[str] = Field(min_length=1)
    notes: str | None = None

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("relevant_source_ids")
    @classmethod
    def validate_source_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("relevant source IDs must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("relevant source IDs must be unique")
        return normalized


class ModeMetrics(BaseModel):
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    query_count: int = Field(ge=0)


class RetrievedDiagnostic(BaseModel):
    source_id: str
    chunk_id: str
    rank: int = Field(ge=1)
    score: float
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class QueryDiagnostic(BaseModel):
    mode: RetrievalMode
    repository: str
    query: str
    relevant_source_ids: list[str]
    retrieved: list[RetrievedDiagnostic]
    first_relevant_rank: int | None
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_10: float
    succeeded_at_5: bool
    succeeded_at_10: bool


class EvaluationConfig(BaseModel):
    embedding_model: str
    reranker_model: str
    rrf_k: int
    candidates_per_backend: int
    rerank_candidate_pool: int
    retrieval_depth: int


class EvaluationReport(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_path: str
    repositories: list[str]
    dataset_size: int
    config: EvaluationConfig
    metrics: dict[RetrievalMode, ModeMetrics]
    diagnostics: list[QueryDiagnostic]
