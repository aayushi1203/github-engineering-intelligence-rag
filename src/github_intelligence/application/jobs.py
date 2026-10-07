"""PostgreSQL is the queue; lease grants fence every mutable job operation."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, exists, func, or_, select, update
from sqlalchemy.orm import Session, aliased

from github_intelligence.indexing.models import IndexHead, IndexOperation
from github_intelligence.persistence.models import IngestionJob, Repository, WebhookDelivery
from github_intelligence.persistence.operations import JobStatus, create_job

_ACTIVE = ["FETCHING", "PARSING", "EMBEDDING", "INDEXING"]
_ORDER = [status.value for status in JobStatus if status != JobStatus.FAILED]


class LeaseLost(RuntimeError):
    pass


def enqueue(
    session: Session, repository_id: UUID, *, key: str, scope: dict[str, object]
) -> IngestionJob:
    return create_job(
        session, repository_id, operation="github_ingest", scope=scope, idempotency_key=key
    )


def claim(session: Session, *, lease_seconds: int, max_attempts: int) -> IngestionJob | None:
    now = session.scalar(select(func.now()))
    assert now is not None
    eligible = or_(
        IngestionJob.status == "QUEUED",
        and_(
            IngestionJob.status == "FAILED",
            IngestionJob.retryable.is_(True),
            IngestionJob.next_attempt_at <= now,
        ),
        and_(IngestionJob.status.in_(_ACTIVE), IngestionJob.lease_expires_at <= now),
    )
    budget = func.coalesce(IngestionJob.scope["operator_attempt_budget"].as_integer(), max_attempts)
    # Exhausted expired claims become terminal instead of disappearing from status views.
    exhausted = session.scalars(
        select(IngestionJob)
        .where(
            IngestionJob.operation == "github_ingest",
            eligible,
            IngestionJob.attempts >= budget,
        )
        .with_for_update(skip_locked=True)
    ).all()
    for exhausted_job in exhausted:
        exhausted_job.status, exhausted_job.retryable = "FAILED", False
        exhausted_job.error_code, exhausted_job.error_detail = (
            "attempts_exhausted",
            "Worker attempt budget exhausted.",
        )
        exhausted_job.lease_owner = exhausted_job.lease_expires_at = None
        exhausted_job.completed_at = exhausted_job.updated_at = now
        delivery_state(session, exhausted_job.id, "FAILED", "attempts_exhausted")
    earlier = aliased(IngestionJob)
    predecessor = exists(
        select(earlier.id).where(
            earlier.repository_id == IngestionJob.repository_id,
            earlier.operation == "github_ingest",
            or_(
                earlier.status.in_(["QUEUED", *_ACTIVE]),
                and_(earlier.status == "FAILED", earlier.retryable.is_(True)),
            ),
            or_(
                earlier.created_at < IngestionJob.created_at,
                and_(earlier.created_at == IngestionJob.created_at, earlier.id < IngestionJob.id),
            ),
        )
    )
    job = session.scalars(
        select(IngestionJob)
        .where(
            IngestionJob.operation == "github_ingest",
            eligible,
            IngestionJob.attempts < budget,
            ~predecessor,
        )
        .order_by(IngestionJob.created_at, IngestionJob.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    ).one_or_none()
    if job is None:
        return None
    # Serialize claims with receipt enqueue/coalescing, even across timestamp ties.
    if (
        session.scalar(
            select(Repository.id)
            .where(Repository.id == job.repository_id)
            .with_for_update(skip_locked=True)
        )
        is None
    ):
        return None
    other_active = session.scalar(
        select(IngestionJob.id)
        .where(
            IngestionJob.repository_id == job.repository_id,
            IngestionJob.id != job.id,
            IngestionJob.status.in_(_ACTIVE),
        )
        .limit(1)
    )
    if other_active is not None:
        return None
    job.lease_owner = uuid4()  # New grant, not a reusable process identity.
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    job.heartbeat_at, job.updated_at = now, now
    job.status = "FETCHING"
    job.attempts += 1
    job.started_at = job.started_at or now
    job.completed_at = None
    job.error_code = job.error_detail = None
    job.retryable, job.next_attempt_at = False, None
    session.flush()
    delivery_state(session, job.id, "PROCESSING")
    return job


def owned(session: Session, job_id: UUID, owner: UUID) -> IngestionJob:
    job = session.scalars(
        select(IngestionJob)
        .where(
            IngestionJob.id == job_id,
            IngestionJob.lease_owner == owner,
            IngestionJob.status.in_(_ACTIVE),
            IngestionJob.lease_expires_at > func.clock_timestamp(),
        )
        .with_for_update()
    ).one_or_none()
    if job is None:
        raise LeaseLost("Job lease is no longer owned")
    return job


def heartbeat(session: Session, job_id: UUID, owner: UUID, *, lease_seconds: int) -> None:
    job = owned(session, job_id, owner)
    now = session.scalar(select(func.clock_timestamp()))
    assert now is not None
    job.heartbeat_at, job.lease_expires_at = now, now + timedelta(seconds=lease_seconds)


def advance(
    session: Session, job_id: UUID, owner: UUID, status: str, *, index_stage: str | None = None
) -> None:
    job = owned(session, job_id, owner)
    if status not in _ORDER or status == "COMPLETE":
        raise ValueError("Use completion validation for terminal stage")
    if _ORDER.index(status) < _ORDER.index(job.status):
        raise ValueError("Job stages cannot move backwards within an attempt")
    if _ORDER.index(status) > _ORDER.index(job.status) + 1:
        raise ValueError("Job stages cannot skip a public lifecycle stage")
    job.status, job.index_stage = status, index_stage
    job.updated_at = datetime.now(UTC)


def complete(session: Session, job_id: UUID, owner: UUID, revision: UUID) -> None:
    job = owned(session, job_id, owner)
    operation = session.get(IndexOperation, revision)
    head = session.get(IndexHead, job.repository_id)
    if (
        job.status != "INDEXING"
        or operation is None
        or operation.status != "READY"
        or operation.repository_id != job.repository_id
        or head is None
    ):
        raise ValueError("COMPLETE requires the current verified READY revision")
    job.status, job.result_revision_id = "COMPLETE", revision
    job.completed_at, job.updated_at = datetime.now(UTC), datetime.now(UTC)
    job.lease_owner = job.lease_expires_at = None
    job.retryable, job.next_attempt_at = False, None
    job.error_code = job.error_detail = None
    delivery_state(session, job.id, "COMPLETE")


def fail(
    session: Session,
    job_id: UUID,
    owner: UUID,
    *,
    code: str,
    retryable: bool,
    max_attempts: int,
    delay_seconds: float,
) -> None:
    job = owned(session, job_id, owner)
    now = session.scalar(select(func.clock_timestamp()))
    assert now is not None
    job.status = "FAILED"
    job.error_code, job.error_detail = (
        code,
        "Ingestion attempt failed; see the sanitized failure code.",
    )
    job.retryable = retryable and job.attempts < max_attempts
    job.next_attempt_at = now + timedelta(seconds=delay_seconds) if job.retryable else None
    job.completed_at, job.updated_at = now, now
    job.lease_owner = job.lease_expires_at = None

    delivery_state(session, job.id, "FAILED", code)


def delivery_state(session: Session, job_id: UUID, status: str, code: str | None = None) -> None:
    session.execute(
        update(WebhookDelivery)
        .where(WebhookDelivery.job_id == job_id)
        .values(status=status, error_code=code)
    )


def requeue(session: Session, job_id: UUID, *, additional_attempts: int) -> IngestionJob:
    """Explicit operator recovery after repair; preserve snapshots, plans and lifetime attempts."""
    job = session.scalars(
        select(IngestionJob).where(IngestionJob.id == job_id).with_for_update()
    ).one_or_none()
    if job is None:
        raise LookupError("Unknown job")
    if job.operation != "github_ingest" or job.status != "FAILED" or job.retryable:
        raise ValueError("Only terminal failed ingestion can be explicitly retried")
    if not 1 <= additional_attempts <= 10:
        raise ValueError("Invalid operator retry budget")
    job.scope = {**job.scope, "operator_attempt_budget": job.attempts + additional_attempts}
    job.status, job.completed_at = "QUEUED", None
    job.next_attempt_at = None
    job.error_code = job.error_detail = None
    job.updated_at = datetime.now(UTC)
    delivery_state(session, job.id, "ACCEPTED")
    return job
