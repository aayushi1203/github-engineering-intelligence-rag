from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from github_intelligence.models import DocumentType, GitHubDocument, RetrievalResult


class ToolEvidence(BaseModel):
    """Citation-eligible evidence returned by every GitHub MCP tool."""

    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    text: str
    title: str
    source_url: HttpUrl
    source_type: DocumentType
    repository: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_document(cls, document: GitHubDocument) -> "ToolEvidence":
        return cls(
            evidence_id=(
                f"github:{document.repository}:{document.document_type.value}:{document.github_id}"
            ),
            text=document.text,
            title=document.title,
            source_url=document.source_url,
            source_type=document.document_type,
            repository=document.repository,
            metadata=document.metadata | {"github_id": document.github_id},
        )

    @classmethod
    def from_retrieval(cls, result: RetrievalResult) -> "ToolEvidence":
        return cls(
            evidence_id=result.chunk_id,
            text=result.text,
            title=result.title,
            source_url=result.source_url,
            source_type=result.source_type,
            repository=result.repository,
            metadata=result.metadata,
        )

    def as_retrieval_result(self) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=self.evidence_id,
            text=self.text,
            title=self.title,
            source_url=self.source_url,
            source_type=self.source_type,
            score=0.0,
            repository=self.repository,
            metadata=self.metadata,
        )


class ToolEvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    repository: str
    items: list[ToolEvidence] = Field(default_factory=list)
    message: str | None = None
