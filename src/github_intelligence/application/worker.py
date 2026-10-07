"""Explicit worker process: durable leases, fenced publication and bounded retries."""

import asyncio
import os
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from github_intelligence.application import jobs
from github_intelligence.application.failures import ControlledFailure, Failure, classify
from github_intelligence.application.runtime import Runtime
from github_intelligence.indexing.coordinator import IndexingError
from github_intelligence.ingestion import IngestionOptions, RepositoryIngestor
from github_intelligence.ingestion.snapshot import (
    load_snapshot_report,
    publish_snapshot,
    report_path,
)
from github_intelligence.persistence.models import Repository


class ShutdownRequested(ControlledFailure):
    def __init__(self, message: str):
        super().__init__("worker_shutdown", True)


class Worker:
    def __init__(self, runtime: Runtime, *, boundary: Callable[[str], None] | None = None):
        self.runtime = runtime
        self.stop = threading.Event()
        self.boundary = boundary or (lambda _: None)

    def _stage(self, job_id: UUID, owner: UUID, stage: str, actual: str) -> None:
        with Session(self.runtime.engine) as session, session.begin():
            jobs.advance(session, job_id, owner, stage, index_stage=actual)

    async def _fetch(
        self, name: str, path: Path, options: IngestionOptions, lost: threading.Event
    ) -> None:
        client = self.runtime.github_client()
        try:
            task = asyncio.create_task(
                publish_snapshot(
                    RepositoryIngestor(client, self.runtime.settings.github_doc_max_bytes),
                    name,
                    path,
                    options=options,
                )
            )
            try:
                while not task.done():
                    if self.stop.is_set() or lost.is_set():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        raise ShutdownRequested("Fetch interrupted")
                    await asyncio.sleep(0.1)
                await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        finally:
            await client.aclose()

    def run_one(self) -> bool:
        settings = self.runtime.settings
        with Session(self.runtime.engine, expire_on_commit=False) as session, session.begin():
            job = jobs.claim(
                session,
                lease_seconds=settings.worker_lease_seconds,
                max_attempts=settings.worker_max_attempts,
            )
            if job is None:
                return False
            job_id, owner, scope, attempt = job.id, job.lease_owner, job.scope, job.attempts
            attempt_budget = int(scope.get("operator_attempt_budget", settings.worker_max_attempts))
            repository = session.get(Repository, job.repository_id)
            assert repository is not None and owner is not None
            name = repository.name
        finished, lost = threading.Event(), threading.Event()

        def renew() -> None:
            while not finished.wait(settings.worker_heartbeat_seconds):
                try:
                    with Session(self.runtime.engine) as session, session.begin():
                        jobs.heartbeat(
                            session, job_id, owner, lease_seconds=settings.worker_lease_seconds
                        )
                except Exception:
                    lost.set()  # Fail closed on lost lease or inability to verify ownership.
                    return

        keeper = threading.Thread(target=renew, daemon=True)
        keeper.start()
        directory = settings.application_root / "snapshots" / str(job_id)
        canonical = directory / "snapshot.jsonl"
        temporary = directory / str(owner) / "snapshot.jsonl"
        try:
            if scope.get("configuration") != self.runtime.configuration():
                raise ValueError("Job runtime configuration changed")
            self._stage(
                job_id,
                owner,
                "FETCHING",
                "RECOVERING_SNAPSHOT" if canonical.exists() else "FETCHING_AND_NORMALIZING",
            )
            if canonical.exists():
                load_snapshot_report(canonical)
            else:
                options = IngestionOptions(**scope["options"])
                asyncio.run(self._fetch(name, temporary, options, lost))
                load_snapshot_report(temporary)
                # Only an owned grant can install canonical files. A stale worker's
                # temporary files cannot overwrite another attempt's canonical snapshot.
                with Session(self.runtime.engine) as session, session.begin():
                    jobs.owned(session, job_id, owner)
                    directory.mkdir(parents=True, exist_ok=True)
                    if not canonical.exists():
                        os.replace(report_path(temporary), report_path(canonical))
                        os.replace(temporary, canonical)
            report = load_snapshot_report(canonical)
            self._stage(job_id, owner, "PARSING", "VALIDATING_SNAPSHOT")
            with Session(self.runtime.engine) as session, session.begin():
                active = jobs.owned(session, job_id, owner)
                active.snapshot_path = str(canonical.resolve())
                active.total_count = active.processed_count = sum(
                    scan.documents_emitted for scan in report.sources.values()
                )
            self.boundary("after_ingestion")
            self._stage(job_id, owner, "EMBEDDING", "PREPARING")

            def checkpoint(stage: str) -> None:
                if self.stop.is_set():
                    raise ShutdownRequested("Worker is shutting down")
                if lost.is_set():
                    raise jobs.LeaseLost("Lease extension failed")
                if stage == "publishing":
                    self.boundary(stage)
                    return  # Publication guard already holds the job row in this transaction.
                if stage in {"prepared", "before_vector_writes", "vector_writes_started"}:
                    self._stage(job_id, owner, "EMBEDDING", "VECTOR_WRITES")
                elif stage in {"vectors_written", "lexical_built", "verified"}:
                    self._stage(job_id, owner, "INDEXING", stage.upper())
                else:
                    with Session(self.runtime.engine) as session, session.begin():
                        jobs.owned(session, job_id, owner)
                self.boundary(stage)

            def publication_guard(session: Session, revision: UUID) -> None:
                jobs.owned(
                    session, job_id, owner
                ).result_revision_id = revision  # Same transaction as READY publication.

            operation = self.runtime.coordinator(boundary=checkpoint, guard=publication_guard).run(
                canonical,
                repository=name,
                reconcile=bool(scope["reconcile"]),
                webhook_job_id=job_id if "webhook_target" in scope else None,
            )
            # READY replay does not emit pipeline stages; record their no-op completion.
            self._stage(job_id, owner, "INDEXING", "READY")
            with Session(self.runtime.engine) as session, session.begin():
                jobs.owned(session, job_id, owner).result_revision_id = operation.id
            self.boundary("ready_produced")
            with Session(self.runtime.engine) as session, session.begin():
                jobs.complete(session, job_id, owner, operation.id)
        except Exception as error:
            if isinstance(error, IndexingError):
                failure = Failure(error.code, error.retryable)
            elif isinstance(error, ShutdownRequested):
                failure = Failure("worker_shutdown", True)
            else:
                failure = classify(error)
            delay = max(
                failure.minimum_delay,
                min(
                    settings.worker_retry_max_seconds,
                    settings.worker_retry_base_seconds * 2 ** (attempt - 1),
                ),
            )
            try:
                with Session(self.runtime.engine) as session, session.begin():
                    jobs.fail(
                        session,
                        job_id,
                        owner,
                        code=failure.code,
                        retryable=failure.retryable,
                        max_attempts=attempt_budget,
                        delay_seconds=delay,
                    )
            except Exception:
                pass  # Do not mutate another grant. Expiry recovers DB outages.
        finally:
            finished.set()
            keeper.join(timeout=settings.worker_heartbeat_seconds + 1)
            shutil.rmtree(temporary.parent, ignore_errors=True)
        return True

    def run(self) -> None:
        def loop() -> None:
            while not self.stop.is_set():
                try:
                    available = self.run_one()
                except Exception:
                    available = False  # Database unavailable; bounded poll, no raw log.
                if not available:
                    self.stop.wait(self.runtime.settings.worker_poll_seconds)

        with ThreadPoolExecutor(max_workers=self.runtime.settings.worker_concurrency) as pool:
            futures = [pool.submit(loop) for _ in range(self.runtime.settings.worker_concurrency)]
            for future in futures:
                future.result()
