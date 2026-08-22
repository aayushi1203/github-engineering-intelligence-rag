from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class DocumentType(str, Enum):  # noqa: UP042
    ISSUE = "issue"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_COMMENT = "pull_request_comment"
    PULL_REQUEST_REVIEW_COMMENT = "pull_request_review_comment"
    COMMIT = "commit"
    RELEASE = "release"
    DOCUMENTATION = "documentation"


class GitHubDocument(BaseModel):
    """Canonical representation of a retrievable GitHub source."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    document_type: DocumentType
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    title: str = Field(min_length=1)
    source_url: HttpUrl
    github_id: str = Field(min_length=1)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", "title")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value
