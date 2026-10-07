"""Application tables; schema creation is exclusively managed by Alembic."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Identity:
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4, server_default=func.gen_random_uuid()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class User(Identity, Base):
    __tablename__ = "users"
    external_identity: Mapped[str | None] = mapped_column(Text, unique=True)


class Repository(Identity, Base):
    __tablename__ = "repositories"
    __table_args__ = (
        CheckConstraint(
            "name = lower(name) AND name ~ '^[a-z0-9][a-z0-9-]*/[a-z0-9_.-]+$'",
            name="ck_repository_name",
        ),
        CheckConstraint(
            "visibility IN ('unknown', 'public', 'private', 'internal')",
            name="ck_repository_visibility",
        ),
    )
    github_id: Mapped[str | None] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    github_url: Mapped[str] = mapped_column(Text)
    default_branch: Mapped[str | None] = mapped_column(Text)
    visibility: Mapped[str] = mapped_column(String(16), server_default="unknown")
    owner_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Identity, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("repository_id", "source_type", "source_key", name="uq_document_source"),
        CheckConstraint(
            "source_type IN ('issue','issue_comment','pull_request',"
            "'pull_request_comment','pull_request_review_comment','pull_request_review',"
            "'commit','release','documentation')",
            name="ck_document_source_type",
        ),
        CheckConstraint(
            "index_status IN ('pending','indexed','failed','deleted')",
            name="ck_document_index_status",
        ),
        CheckConstraint(
            "content_hash ~ '^sha256:normalized-v1:[0-9a-f]{64}$'", name="ck_document_hash"
        ),
        Index("ix_documents_repository_status", "repository_id", "index_status"),
    )
    repository_id: Mapped[UUID] = mapped_column(ForeignKey("repositories.id"))
    source_type: Mapped[str] = mapped_column(String(40))
    source_key: Mapped[str] = mapped_column(Text)
    github_id: Mapped[str] = mapped_column(Text)
    github_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    source_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    commit_sha: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    embedding_version: Mapped[str | None] = mapped_column(Text)
    chunker_version: Mapped[str | None] = mapped_column(Text)
    index_status: Mapped[str] = mapped_column(String(16), server_default="pending")
    normalized_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestionJob(Identity, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("repository_id", "operation", "idempotency_key", name="uq_job_request"),
        CheckConstraint(
            "status IN ('QUEUED','FETCHING','PARSING','EMBEDDING','INDEXING','COMPLETE','FAILED')",
            name="ck_job_status",
        ),
        CheckConstraint(
            "processed_count >= 0 AND (total_count IS NULL OR total_count >= processed_count) "
            "AND attempts >= 0",
            name="ck_job_progress",
        ),
        Index("ix_ingestion_jobs_status_created", "status", "created_at"),
    )
    repository_id: Mapped[UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    lease_owner: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retryable: Mapped[bool] = mapped_column(server_default="false")
    result_revision_id: Mapped[UUID | None] = mapped_column(ForeignKey("index_operations.id"))
    snapshot_path: Mapped[str | None] = mapped_column(Text)
    index_stage: Mapped[str | None] = mapped_column(Text)
    requested_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    operation: Mapped[str] = mapped_column(Text)
    scope: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    idempotency_key: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), server_default="QUEUED")
    processed_count: Mapped[int] = mapped_column(Integer, server_default="0")
    total_count: Mapped[int | None] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    error_code: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[str | None] = mapped_column(Text)


class Query(Identity, Base):
    __tablename__ = "queries"
    __table_args__ = (
        UniqueConstraint("repository_id", "request_key", name="uq_query_request"),
        CheckConstraint(
            "status IN ('pending','complete','failed','insufficient_evidence')",
            name="ck_query_status",
        ),
    )
    repository_id: Mapped[UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    request_key: Mapped[str | None] = mapped_column(Text)
    correlation_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), default=uuid4)
    status: Mapped[str] = mapped_column(String(24), server_default="pending")
    model: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(Text)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    pipeline_version: Mapped[str | None] = mapped_column(Text)
    measurements: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    query_text: Mapped[str | None] = mapped_column(Text)


class Answer(Identity, Base):
    __tablename__ = "answers"
    __table_args__ = (
        UniqueConstraint("query_id", "attempt", name="uq_answer_attempt"),
        CheckConstraint("attempt >= 1", name="ck_answer_attempt"),
        CheckConstraint(
            "status IN ('complete','failed','insufficient_evidence')", name="ck_answer_status"
        ),
        CheckConstraint(
            "citation_validation IN ('valid','invalid','not_evaluated')", name="ck_answer_citations"
        ),
    )
    query_id: Mapped[UUID] = mapped_column(ForeignKey("queries.id"))
    attempt: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24))
    answer_text: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    source_references: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    citation_validation: Mapped[str] = mapped_column(String(16), server_default="not_evaluated")
    limitations: Mapped[list[str]] = mapped_column(JSONB, server_default="[]")


class Feedback(Identity, Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("rating IN ('positive','negative')", name="ck_feedback_rating"),
        CheckConstraint(
            "category IS NULL OR category IN ('incorrect_answer','bad_sources','missing_context',"
            "'unsupported_claim','other')",
            name="ck_feedback_category",
        ),
    )
    answer_id: Mapped[UUID] = mapped_column(ForeignKey("answers.id"), index=True)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    rating: Mapped[str] = mapped_column(String(16))
    category: Mapped[str | None] = mapped_column(String(32))


class WebhookDelivery(Base):
    """Authenticated receipt metadata only; the raw body and headers are never retained."""

    __tablename__ = "webhook_deliveries"
    __table_args__ = (
        CheckConstraint(
            "status IN ('ACCEPTED','PROCESSING','COMPLETE','FAILED','IGNORED','DEFERRED')",
            name="ck_webhook_status",
        ),
    )
    delivery_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64))
    action: Mapped[str | None] = mapped_column(String(64))
    repository_id: Mapped[UUID | None] = mapped_column(ForeignKey("repositories.id"), index=True)
    repository_github_id: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(16))
    job_id: Mapped[UUID | None] = mapped_column(ForeignKey("ingestion_jobs.id"), index=True)
    body_sha256: Mapped[str] = mapped_column(String(64))
    event_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    error_code: Mapped[str | None] = mapped_column(String(64))
