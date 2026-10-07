"""Trusted-local HTTP boundary; requests enqueue durable work, never run ingestion."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from github_intelligence.application.jobs import enqueue
from github_intelligence.application.runtime import Runtime
from github_intelligence.config import Settings
from github_intelligence.generation.provider import LLMProviderError
from github_intelligence.generation.service import CitationValidationError
from github_intelligence.indexing.coordinator import IndexingError
from github_intelligence.indexing.models import IndexHead, IndexOperation
from github_intelligence.persistence.models import IngestionJob, Repository
from github_intelligence.persistence.operations import (
    SourceReference,
    create_answer,
    create_query,
    register_repository,
)


class RepositoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=3, max_length=200)


class IngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    include_comments: bool = True
    include_documentation: bool = True
    include_reviews: bool = True
    reconcile: bool = False


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=8000)
    store_content: bool = False


def error(status: int, code: str) -> HTTPException:
    return HTTPException(status, detail={"code": code})


def repository_in(session: Session, identity: UUID) -> Repository:
    repo = session.get(Repository, identity)
    if repo is None:
        raise error(404, "repository_not_found")
    return repo


def job_view(job: IngestionJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "repository_id": job.repository_id,
        "status": job.status,
        "index_stage": job.index_stage,
        "attempts": job.attempts,
        "processed_count": job.processed_count,
        "total_count": job.total_count,
        "progress_unit": "documents_normalized",
        "retryable": job.retryable,
        "next_attempt_at": job.next_attempt_at,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
        "heartbeat_at": job.heartbeat_at,
        "lease_expires_at": job.lease_expires_at,
        "error_code": job.error_code,
        "error_detail": job.error_detail,
        "result_revision_id": job.result_revision_id,
    }


def create_app(settings: Settings | None = None, *, runtime: Runtime | None = None) -> FastAPI:
    app_runtime = runtime or Runtime(settings or Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if runtime is None:
            app_runtime.close()

    app = FastAPI(title="GitHub Engineering Intelligence", lifespan=lifespan)
    app.state.runtime = app_runtime

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": {"code": "invalid_request"}})

    @app.exception_handler(SQLAlchemyError)
    async def database_failure(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": {"code": "database_unavailable"}})

    @app.post("/webhooks/github")
    async def webhook(request: Request) -> JSONResponse:
        from github_intelligence.application.webhooks import receive, verify_signature

        settings = app_runtime.settings
        if not settings.github_webhooks_enabled or not settings.github_webhook_secret:
            raise error(503, "webhook_unavailable")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > settings.github_webhook_max_body_bytes:
                raise error(413, "webhook_body_too_large")
            raw.extend(chunk)
        if not verify_signature(
            bytes(raw), request.headers.get("x-hub-signature-256"), settings.github_webhook_secret
        ):
            raise error(401, "invalid_webhook_signature")
        try:
            delivery_id = str(UUID(request.headers.get("x-github-delivery", "")))
            event = request.headers.get("x-github-event", "")
            if not event or len(event) > 64 or not all(c.islower() or c == "_" for c in event):
                raise ValueError("Invalid event header")
        except ValueError:
            raise error(400, "invalid_webhook_headers") from None

        def persist() -> tuple[dict[str, Any], int]:
            with Session(app_runtime.engine) as session, session.begin():
                session.execute(text("SET LOCAL lock_timeout = '1000ms'"))
                session.execute(text("SET LOCAL statement_timeout = '3000ms'"))
                delivery, duplicate = receive(
                    session,
                    delivery_id=delivery_id,
                    event=event,
                    raw=bytes(raw),
                    configuration=app_runtime.configuration(),
                )
                response = {
                    "delivery_id": delivery.delivery_id,
                    "status": delivery.status,
                    "duplicate": duplicate,
                    "job_id": str(delivery.job_id) if delivery.job_id else None,
                    "code": delivery.error_code,
                }
                status = (
                    200
                    if duplicate or delivery.status in {"IGNORED", "DEFERRED"}
                    else 400
                    if delivery.status == "FAILED"
                    else 202
                )
            return response, status  # Commit before acknowledgement.

        try:
            response, status = await run_in_threadpool(persist)
        except ValueError:
            raise error(409, "webhook_delivery_conflict") from None
        return JSONResponse(status_code=status, content=response)

    @app.get("/webhook-deliveries/{delivery_id}")
    def delivery_status(delivery_id: UUID) -> dict[str, Any]:
        from github_intelligence.persistence.models import WebhookDelivery

        with Session(app_runtime.engine) as session:
            delivery = session.get(WebhookDelivery, str(delivery_id))
            if delivery is None:
                raise error(404, "webhook_delivery_not_found")
            job = session.get(IngestionJob, delivery.job_id) if delivery.job_id else None
            return {
                "delivery_id": delivery.delivery_id,
                "status": delivery.status,
                "received_at": delivery.received_at,
                "event_type": delivery.event_type,
                "action": delivery.action,
                "metadata": delivery.event_metadata,
                "code": delivery.error_code,
                "job": job_view(job) if job else None,
            }

    @app.post("/repositories", status_code=201)
    def register(body: RepositoryRequest) -> dict[str, Any]:
        try:
            with Session(app_runtime.engine) as session, session.begin():
                repo = register_repository(session, body.name)
                return {"id": repo.id, "name": repo.name, "github_url": repo.github_url}
        except ValueError:
            raise error(422, "invalid_repository") from None

    @app.get("/repositories/{repository_id}")
    def repository(repository_id: UUID) -> dict[str, Any]:
        with Session(app_runtime.engine) as session:
            repo = repository_in(session, repository_id)
            head = session.get(IndexHead, repository_id)
            operation = session.get(IndexOperation, head.revision_id) if head else None
            return {
                "id": repo.id,
                "name": repo.name,
                "github_id": repo.github_id,
                "github_url": repo.github_url,
                "default_branch": repo.default_branch,
                "visibility": repo.visibility,
                "index_status": operation.status if operation else "NOT_INDEXED",
                "ready_revision": head.revision_id if head else None,
            }

    @app.post("/repositories/{repository_id}/ingestions", status_code=202)
    def ingest(repository_id: UUID, body: IngestionRequest) -> dict[str, Any]:
        try:
            with Session(app_runtime.engine) as session, session.begin():
                repository_in(session, repository_id)
                job = enqueue(
                    session,
                    repository_id,
                    key=body.idempotency_key,
                    scope={
                        "configuration": app_runtime.configuration(),
                        "reconcile": body.reconcile,
                        "options": {
                            "include_comments": body.include_comments,
                            "include_documentation": body.include_documentation,
                            "include_reviews": body.include_reviews,
                            "fresh": True,
                        },
                    },
                )
                return job_view(job)
        except ValueError:
            raise error(409, "idempotency_conflict") from None

    @app.post("/ingestion-jobs/{job_id}/retry", status_code=202)
    def retry_ingestion(job_id: UUID) -> dict[str, Any]:
        from github_intelligence.application.jobs import requeue

        try:
            with Session(app_runtime.engine) as session, session.begin():
                job = requeue(
                    session, job_id, additional_attempts=app_runtime.settings.worker_max_attempts
                )
                return job_view(job)
        except LookupError:
            raise error(404, "ingestion_job_not_found") from None
        except ValueError:
            raise error(409, "ingestion_job_not_terminal_failed") from None

    @app.get("/ingestion-jobs/{job_id}")
    def job_status(job_id: UUID) -> dict[str, Any]:
        with Session(app_runtime.engine) as session:
            job = session.get(IngestionJob, job_id)
            if job is None:
                raise error(404, "ingestion_job_not_found")
            return job_view(job)

    @app.post("/repositories/{repository_id}/query")
    def query(repository_id: UUID, body: QueryRequest) -> dict[str, Any]:
        if not body.question.strip():
            raise error(422, "invalid_request")
        with Session(app_runtime.engine) as session:
            name = repository_in(session, repository_id).name
            if session.get(IndexHead, repository_id) is None:
                raise error(409, "repository_not_indexed")
        with Session(app_runtime.engine) as session, session.begin():
            record = create_query(
                session,
                repository_id,
                query_text=body.question,
                store_content=body.store_content,
                model=app_runtime.settings.openai_model,
                provider="openai",
            )
            query_id = record.id
        try:
            # A fresh immutable READY view per request prevents stale cached BM25 state.
            ready = app_runtime.coordinator().ready(name)
            service = app_runtime.answers(ready)
            try:
                from github_intelligence.retrieval import RetrievalMode

                results = service.pipeline.search(
                    body.question,
                    mode=RetrievalMode.HYBRID_RERANK,
                    top_k=service.top_k,
                    repository=name,
                )
            except Exception:
                raise error(503, "retrieval_infrastructure_unavailable") from None
            try:
                answer = service.answer_from_results(
                    body.question, repository=name, results=results
                )
            except (LLMProviderError, CitationValidationError, ValueError):
                raise error(502, "generation_failed") from None
            status: Literal["complete", "insufficient_evidence"] = (
                "complete" if answer.citations else "insufficient_evidence"
            )
            with Session(app_runtime.engine) as session, session.begin():
                query_record = session.get(type(record), query_id)
                assert query_record is not None
                query_record.status = status
                query_record.configuration = {"index_revision": str(ready.revision)}
                query_record.pipeline_version = "ready-hybrid-v1"
                result = create_answer(
                    session,
                    query_id,
                    status=status,
                    answer_text=answer.answer,
                    store_content=body.store_content,
                    citations=[
                        SourceReference(source_id=c.source_id, github_url=c.source_url)
                        for c in answer.citations
                    ],
                    sources=[
                        SourceReference(
                            source_id=s.source_id, github_url=s.source_url, chunk_id=s.chunk_id
                        )
                        for s in answer.retrieved_sources
                    ],
                    citation_validation="valid",
                    limitations=answer.limitations or [],
                )
                answer_id = result.id
            return {
                "query_id": query_id,
                "answer_id": answer_id,
                "status": status,
                "revision": ready.revision,
                **answer.model_dump(mode="json"),
            }
        except IndexingError:
            failure = error(503, "evidence_unavailable")
        except HTTPException as exc:
            failure = exc
        except ValueError:
            failure = error(503, "generation_not_configured")
        except Exception:
            failure = error(503, "evidence_unavailable")
        with Session(app_runtime.engine) as session, session.begin():
            query_record = session.get(type(record), query_id)
            assert query_record is not None
            query_record.status = "failed"
        raise failure

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    def health_ready() -> dict[str, str]:
        try:
            with app_runtime.engine.connect() as connection:
                connection.execute(text("SET LOCAL statement_timeout = '2000ms'"))
                connection.execute(text("SELECT 1"))
                connection.execute(
                    select(IngestionJob.lease_owner, IngestionJob.result_revision_id).limit(0)
                )
                connection.execute(select(IndexHead.revision_id).limit(0))
                if app_runtime.settings.github_webhooks_enabled:
                    from github_intelligence.persistence.models import WebhookDelivery

                    connection.execute(select(WebhookDelivery.delivery_id).limit(0))
            app_runtime.store.client.get_collections()
            if not app_runtime.settings.application_root.is_dir():
                raise ValueError("Shared revision directory unavailable")
            if app_runtime.provider is None and not app_runtime.settings.openai_api_key:
                raise ValueError("Generation not configured")
        except Exception:
            raise error(503, "application_not_ready") from None
        return {"status": "ready"}

    return app
