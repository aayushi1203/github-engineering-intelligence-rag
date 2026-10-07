"""Actual PostgreSQL row locks and lease grants, not mocked queue behavior."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event

import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from github_intelligence.application import jobs
from github_intelligence.persistence.models import IngestionJob
from github_intelligence.persistence.operations import register_repository


def queued(engine, name="acme/repo", key="request"):
    with Session(engine, expire_on_commit=False) as session, session.begin():
        repo = register_repository(session, name)
        return jobs.enqueue(session, repo.id, key=key, scope={"configuration": {}})


def take(engine, max_attempts=3):
    with Session(engine, expire_on_commit=False) as session, session.begin():
        return jobs.claim(session, lease_seconds=30, max_attempts=max_attempts)


def expire(engine, identity):
    with Session(engine) as session, session.begin():
        value = session.get(IngestionJob, identity)
        value.lease_expires_at = session.scalar(select(func.now())) - timedelta(seconds=1)


def test_creation_is_idempotent_and_conflicting_scope_rejected(migrated_engine):
    first = queued(migrated_engine)
    second = queued(migrated_engine)
    assert first.id == second.id
    with Session(migrated_engine) as session, session.begin():
        with pytest.raises(ValueError, match="different"):
            jobs.enqueue(session, first.repository_id, key="request", scope={"different": True})
    with Session(migrated_engine) as session:
        assert len(session.scalars(select(IngestionJob)).all()) == 1


def test_concurrent_claim_uses_skip_locked_and_never_steals_valid_lease(migrated_engine):
    first = queued(migrated_engine)
    held, release = Event(), Event()

    def owner():
        with Session(migrated_engine, expire_on_commit=False) as session, session.begin():
            result = jobs.claim(session, lease_seconds=30, max_attempts=3)
            held.set()
            assert release.wait(10)
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(owner)
        assert held.wait(10)
        assert take(migrated_engine) is None
        second = queued(migrated_engine, "acme/other")
        independent = take(migrated_engine)
        assert independent.id == second.id
        release.set()
        result = future.result(timeout=10)
    assert result.id == first.id and take(migrated_engine) is None


def test_expiry_reclaim_new_grant_heartbeat_and_stale_owner_fencing(migrated_engine):
    original = queued(migrated_engine)
    one = take(migrated_engine)
    with Session(migrated_engine) as session, session.begin():
        jobs.heartbeat(session, one.id, one.lease_owner, lease_seconds=60)
    with Session(migrated_engine) as session:
        assert session.get(IngestionJob, one.id).lease_expires_at > one.lease_expires_at
    assert take(migrated_engine) is None
    expire(migrated_engine, original.id)
    two = take(migrated_engine)
    assert two.attempts == 2 and two.lease_owner != one.lease_owner
    with Session(migrated_engine) as session, session.begin():
        with pytest.raises(jobs.LeaseLost):
            jobs.owned(session, one.id, one.lease_owner)
        jobs.owned(session, two.id, two.lease_owner)


def test_lifecycle_invalid_transitions_and_complete_requires_ready(migrated_engine):
    queued(migrated_engine)
    active = take(migrated_engine)
    with Session(migrated_engine) as session, session.begin():
        with pytest.raises(ValueError):
            jobs.advance(session, active.id, active.lease_owner, "INDEXING")
        jobs.advance(session, active.id, active.lease_owner, "PARSING")
        jobs.advance(session, active.id, active.lease_owner, "EMBEDDING")
        jobs.advance(session, active.id, active.lease_owner, "INDEXING")
        with pytest.raises(ValueError):
            jobs.advance(session, active.id, active.lease_owner, "FETCHING")
        with pytest.raises(ValueError):
            jobs.complete(session, active.id, active.lease_owner, active.id)


def test_retry_schedule_nonretryable_and_attempt_budget(migrated_engine):
    queued(migrated_engine)
    active = take(migrated_engine, max_attempts=2)
    with Session(migrated_engine) as session, session.begin():
        jobs.fail(
            session,
            active.id,
            active.lease_owner,
            code="github_transient",
            retryable=True,
            max_attempts=2,
            delay_seconds=60,
        )
    assert take(migrated_engine, max_attempts=2) is None
    with Session(migrated_engine) as session, session.begin():
        value = session.get(IngestionJob, active.id)
        value.next_attempt_at = session.scalar(select(func.now())) - timedelta(seconds=1)
    retry = take(migrated_engine, max_attempts=2)
    assert retry.attempts == 2
    expire(migrated_engine, retry.id)
    assert take(migrated_engine, max_attempts=2) is None
    with Session(migrated_engine) as session:
        value = session.get(IngestionJob, retry.id)
        assert value.status == "FAILED" and not value.retryable
        assert value.error_code == "attempts_exhausted"
    queued(migrated_engine, key="permanent")
    terminal = take(migrated_engine)
    with Session(migrated_engine) as session, session.begin():
        jobs.fail(
            session,
            terminal.id,
            terminal.lease_owner,
            code="github_authorization_rejected",
            retryable=False,
            max_attempts=3,
            delay_seconds=0,
        )
    assert take(migrated_engine) is None
