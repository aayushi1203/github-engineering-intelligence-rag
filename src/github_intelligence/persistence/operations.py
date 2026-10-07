"""Small persistence operations; callers own commit/rollback and sessions."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, HttpUrl
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from github_intelligence.indexing import models as index_models  # noqa: F401
from github_intelligence.models import GitHubDocument
from github_intelligence.persistence.identity import (
    canonical_repository,
    content_hash,
    document_id,
    normalized_payload,
    reject_secrets,
    source_key,
)
from github_intelligence.persistence.models import (
    Answer,
    Document,
    Feedback,
    IngestionJob,
    Query,
    Repository,
    User,
)


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    FETCHING = "FETCHING"
    PARSING = "PARSING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class SourceReference(BaseModel):
    """Reference-only persistence; source text belongs to the document payload."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    github_url: HttpUrl
    document_id: UUID | None = None
    chunk_id: str | None = None


@dataclass
class ImportCounts:
    new: int = 0
    changed: int = 0
    unchanged: int = 0


def create_user(session: Session, *, external_identity: str | None = None) -> User:
    user = User(external_identity=external_identity)
    session.add(user)
    session.flush()
    return user


def register_repository(
    session: Session,
    name: str,
    *,
    github_id: str | None = None,
    default_branch: str | None = None,
    visibility: str | None = None,
    owner_user_id: UUID | None = None,
) -> Repository:
    name = canonical_repository(name)
    values: dict[str, Any] = {"name": name, "github_url": f"https://github.com/{name}"}
    for key, value in {
        "github_id": github_id,
        "default_branch": default_branch,
        "visibility": visibility,
        "owner_user_id": owner_user_id,
    }.items():
        if value is not None:
            values[key] = value
    session.execute(
        insert(Repository).values(**values).on_conflict_do_nothing(index_elements=["name"])
    )
    repository = session.scalars(
        select(Repository).where(Repository.name == name).with_for_update()
    ).one()
    changed = False
    for key, value in values.items():
        if getattr(repository, key) != value:
            setattr(repository, key, value)
            changed = True
    if changed:
        repository.updated_at = datetime.now(UTC)
    session.flush()
    return repository


def upsert_document(
    session: Session, repository: Repository, document: GitHubDocument
) -> Literal["new", "changed", "unchanged"]:
    if repository.name != canonical_repository(document.repository):
        raise ValueError("Document repository does not match registered repository")
    # Lock the repository row to serialize metadata writers, including new keys.
    session.execute(select(Repository.id).where(Repository.id == repository.id).with_for_update())
    payload = normalized_payload(document)
    digest = content_hash(document)
    key = source_key(document)
    existing = session.scalars(
        select(Document)
        .where(
            Document.repository_id == repository.id,
            Document.source_type == document.document_type.value,
            Document.source_key == key,
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    ).one_or_none()
    if existing is not None and existing.content_hash == digest and existing.deleted_at is None:
        return "unchanged"
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "github_id": document.github_id,
        "github_url": str(document.source_url),
        "title": document.title,
        "author": document.metadata.get("author"),
        "source_created_at": document.created_at,
        "source_updated_at": document.updated_at,
        "commit_sha": document.metadata.get("commit_sha")
        or document.metadata.get("sha")
        or document.metadata.get("commit_id"),
        "content_hash": digest,
        "normalized_payload": payload,
        "index_status": "pending",
        "last_seen_at": now,
        "updated_at": now,
        "deleted_at": None,
        # These describe actual indexed content, so new imports must not invent versions.
        "embedding_version": None,
        "chunker_version": None,
    }
    if existing is None:
        session.add(
            Document(
                id=document_id(document),
                repository_id=repository.id,
                source_type=document.document_type.value,
                source_key=key,
                **values,
            )
        )
        outcome: Literal["new", "changed", "unchanged"] = "new"
    else:
        for field, value in values.items():
            setattr(existing, field, value)
        outcome = "changed"
    session.flush()
    return outcome


def import_documents(session: Session, documents: Sequence[GitHubDocument]) -> ImportCounts:
    """Validate the entire batch, then upsert in deterministic lock order atomically."""
    for document in documents:
        document_id(document)
        content_hash(document)
    counts = ImportCounts()
    repositories: dict[str, Repository] = {}
    # Prevent cross-repository lock-order deadlocks between imports.
    for name in sorted({canonical_repository(document.repository) for document in documents}):
        repositories[name] = register_repository(session, name)
    for document in documents:
        result = upsert_document(
            session, repositories[canonical_repository(document.repository)], document
        )
        setattr(counts, result, getattr(counts, result) + 1)
    return counts


def create_job(
    session: Session,
    repository_id: UUID,
    *,
    operation: str = "metadata_import",
    scope: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    requested_by: UUID | None = None,
) -> IngestionJob:
    scope = scope or {}
    reject_secrets(scope)
    if not operation.strip() or (idempotency_key is not None and not idempotency_key.strip()):
        raise ValueError("Operation and request keys must not be blank")
    values = dict(
        repository_id=repository_id,
        operation=operation,
        scope=scope,
        idempotency_key=idempotency_key,
        requested_by=requested_by,
    )
    if idempotency_key is None:
        job = IngestionJob(**values)
        session.add(job)
    else:
        session.execute(
            insert(IngestionJob)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_job_request")
        )
        job = session.scalars(
            select(IngestionJob).where(
                IngestionJob.repository_id == repository_id,
                IngestionJob.operation == operation,
                IngestionJob.idempotency_key == idempotency_key,
            )
        ).one()
        if job.scope != scope or job.requested_by != requested_by:
            raise ValueError("Idempotency key already belongs to a different job request")
    session.flush()
    return job


def transition_job(
    session: Session,
    job_id: UUID,
    target: JobStatus,
    *,
    processed_count: int | None = None,
    total_count: int | None = None,
    error_code: str | None = None,
) -> IngestionJob:
    job = session.scalars(
        select(IngestionJob)
        .where(IngestionJob.id == job_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).one()
    order = list(JobStatus)[:-1]
    current = JobStatus(job.status)
    if current in {JobStatus.COMPLETE, JobStatus.FAILED}:
        raise ValueError("Terminal jobs cannot transition; create a new retry job")
    expected = order[order.index(current) + 1]
    if target not in {expected, JobStatus.FAILED}:
        raise ValueError(f"Invalid job transition {current.value} -> {target.value}")
    if error_code is not None and error_code not in {
        "source_unavailable",
        "invalid_document",
        "database_unavailable",
        "index_unavailable",
        "unknown_failure",
    }:
        raise ValueError("Use a sanitized error code, not an exception message")
    if target != JobStatus.FAILED and error_code is not None:
        raise ValueError("Error codes apply only to failed jobs")
    processed = job.processed_count if processed_count is None else processed_count
    total = job.total_count if total_count is None else total_count
    if processed < job.processed_count or (total is not None and total < processed):
        raise ValueError("Invalid job progress")
    now = datetime.now(UTC)
    if target == JobStatus.FETCHING:
        job.started_at = now
        job.attempts += 1
    if target in {JobStatus.COMPLETE, JobStatus.FAILED}:
        job.completed_at = now
    job.status, job.updated_at = target.value, now
    job.processed_count, job.total_count = processed, total
    job.error_code = (error_code or "unknown_failure") if target == JobStatus.FAILED else None
    # Only controlled messages; never raw exception text.
    job.error_detail = (
        "The job failed; consult correlated sanitized logs." if target == JobStatus.FAILED else None
    )
    session.flush()
    return job


def create_query(
    session: Session,
    repository_id: UUID,
    *,
    user_id: UUID | None = None,
    request_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    configuration: dict[str, Any] | None = None,
    pipeline_version: str | None = None,
    measurements: dict[str, Any] | None = None,
    query_text: str | None = None,
    store_content: bool = False,
) -> Query:
    reject_secrets(configuration or {})
    reject_secrets(measurements or {})
    if request_key is not None and not request_key.strip():
        raise ValueError("Request key must not be blank")
    query = Query(
        repository_id=repository_id,
        user_id=user_id,
        request_key=request_key,
        model=model,
        provider=provider,
        configuration=configuration or {},
        pipeline_version=pipeline_version,
        measurements=measurements or {},
        query_text=query_text if store_content else None,
    )
    session.add(query)
    session.flush()
    return query


def create_answer(
    session: Session,
    query_id: UUID,
    *,
    attempt: int = 1,
    status: Literal["complete", "failed", "insufficient_evidence"] = "complete",
    citations: Sequence[SourceReference] = (),
    sources: Sequence[SourceReference] = (),
    citation_validation: Literal["valid", "invalid", "not_evaluated"] = "not_evaluated",
    answer_text: str | None = None,
    limitations: Sequence[str] = (),
    store_content: bool = False,
) -> Answer:
    answer = Answer(
        query_id=query_id,
        attempt=attempt,
        status=status,
        answer_text=answer_text if store_content else None,
        citations=[item.model_dump(mode="json") for item in citations],
        source_references=[item.model_dump(mode="json") for item in sources],
        citation_validation=citation_validation,
        limitations=list(limitations) if store_content else [],
    )
    session.add(answer)
    session.flush()
    return answer


def create_feedback(
    session: Session,
    answer_id: UUID,
    *,
    rating: Literal["positive", "negative"],
    category: Literal[
        "incorrect_answer", "bad_sources", "missing_context", "unsupported_claim", "other"
    ]
    | None = None,
    user_id: UUID | None = None,
) -> Feedback:
    feedback = Feedback(answer_id=answer_id, user_id=user_id, rating=rating, category=category)
    session.add(feedback)
    session.flush()
    return feedback
