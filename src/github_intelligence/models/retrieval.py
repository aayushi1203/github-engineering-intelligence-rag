from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from github_intelligence.models.document import DocumentType


class DocumentChunk(BaseModel):
    """A source-preserving unit used by every retrieval backend."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: HttpUrl
    source_type: DocumentType
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Backend-independent ranked retrieval result."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    title: str
    source_url: HttpUrl
    source_type: DocumentType
    score: float
    repository: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    vector_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    final_rank: int | None = Field(default=None, ge=1)
