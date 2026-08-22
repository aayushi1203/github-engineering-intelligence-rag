from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from github_intelligence.models.document import DocumentType


class AnswerSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^S[1-9]\d*$")
    chunk_id: str
    source_type: DocumentType
    title: str
    source_url: HttpUrl
    repository: str
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^S[1-9]\d*$")
    source_url: HttpUrl


class DraftAnswer(BaseModel):
    """Provider output; source records and URLs are supplied by the application."""

    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    citation_ids: list[str]
    limitations: list[str]


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[Citation]
    sources: list[AnswerSource]
    retrieved_sources: list[AnswerSource]
    limitations: list[str] | None = None


class GenerationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation_validity: float = Field(ge=0, le=1)
    citation_coverage: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    claim_count: int = Field(ge=0)
    cited_claim_count: int = Field(ge=0)
