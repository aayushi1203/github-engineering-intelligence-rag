"""Real PostgreSQL + real server Qdrant; controlled GitHub and inexpensive deterministic models."""

import os
from datetime import timedelta
from uuid import UUID, uuid4

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from github_intelligence.application.api import create_app
from github_intelligence.application.runtime import Runtime
from github_intelligence.application.worker import Worker
from github_intelligence.config import Settings
from github_intelligence.generation.models import DraftAnswer
from github_intelligence.indexing.models import IndexOperation
from github_intelligence.indexing.storage import IncrementalVectorStore
from github_intelligence.persistence.models import Answer, IngestionJob, Query


class Encoder:
    dimension = 3

    def encode(self, texts):
        return [[1.0, float("changed" in text), float("other" in text)] for text in texts]


class Reranker:
    def rerank(self, query, results, *, top_k):
        return list(results)[:top_k]


class Provider:
    def generate(self, **kwargs):
        return DraftAnswer(
            answer="The source describes timeout handling [S1].",
            citation_ids=["S1"],
            limitations=[],
        )


class GitHubBoundary:
    def __init__(self):
        self.count = 0
        self.body = "Timeout handling evidence"
        self.status = None

    def __call__(self, request):
        self.count += 1
        if self.status:
            return httpx.Response(self.status, json={"message": "never-expose-secret-response"})
        path = request.url.path
        name = "/".join(path.split("/")[2:4])
        root = f"/repos/{name}"
        values = {
            root: {
                "id": 42 if name == "acme/repo" else 43,
                "full_name": name,
                "html_url": f"https://github.com/{name}",
                "default_branch": "main",
                "private": False,
            },
            root + "/commits/main": {"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}},
            root + "/issues": [
                {
                    "id": 1,
                    "number": 1,
                    "title": "Timeout",
                    "body": self.body,
                    "html_url": f"https://github.com/{name}/issues/1",
                }
            ],
            root + "/git/trees/" + "b" * 40: {"sha": "b" * 40, "truncated": False, "tree": []},
        }
        return httpx.Response(200, json=values.get(path, []))


@pytest.fixture
def application(migrated_engine, tmp_path):
    url = os.environ.get("TEST_QDRANT_URL")
    if not url:
        pytest.skip("TEST_QDRANT_URL required for actual server-mode integration")
    store = IncrementalVectorStore(None, "phase4_test_" + uuid4().hex, url=url)
    boundary = GitHubBoundary()
    settings = Settings(
        _env_file=None,
        database_url=migrated_engine.url.render_as_string(hide_password=False),
        application_root=tmp_path / "application",
        application_embedding_version="test-v1",
        application_qdrant_url=url,
        worker_heartbeat_seconds=1,
        worker_lease_seconds=30,
        worker_retry_base_seconds=0,
        github_max_retries=1,
    )
    settings.application_root.mkdir()
    runtime = Runtime(
        settings,
        engine=migrated_engine,
        store=store,
        encoder=Encoder(),
        reranker=Reranker(),
        provider=Provider(),
        github_transport=httpx.MockTransport(boundary),
    )
    try:
        yield runtime, boundary
    finally:
        if store.exists():
            store.client.delete_collection(store.collection)
        store.client.close()


def register_and_enqueue(runtime, key="first", **options):
    with TestClient(create_app(runtime=runtime)) as client:
        registered = client.post("/repositories", json={"name": "acme/repo"})
        assert registered.status_code == 201
        repo = registered.json()["id"]
        response = client.post(
            f"/repositories/{repo}/ingestions", json={"idempotency_key": key, **options}
        )
        assert response.status_code == 202
        return repo, response.json()["id"]


def expire(runtime, job_id):
    with Session(runtime.engine) as session, session.begin():
        job = session.get(IngestionJob, UUID(job_id))
        job.lease_expires_at = session.scalar(select(func.now())) - timedelta(seconds=1)


def test_api_queues_without_worker_duplicate_restart_and_query_before_index(application):
    runtime, _ = application
    repo, job = register_and_enqueue(runtime)
    _, duplicate = register_and_enqueue(runtime)
    assert job == duplicate
    with TestClient(create_app(runtime=runtime)) as client:
        result = client.get(f"/ingestion-jobs/{job}")
        assert result.json()["status"] == "QUEUED" and result.json()["attempts"] == 0
        assert (
            client.post(
                f"/repositories/{repo}/query", json={"question": "why timeout?"}
            ).status_code
            == 409
        )
        assert client.get(f"/repositories/{uuid4()}").status_code == 404
        conflict = client.post(
            f"/repositories/{repo}/ingestions", json={"idempotency_key": "first", "reconcile": True}
        )
        assert conflict.status_code == 409
        invalid = client.post("/repositories", json={"name": "bad", "token": "never-echo-token"})
        assert invalid.status_code == 422 and "never-echo" not in invalid.text
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200


def test_worker_without_api_completes_and_queries_ready_retention_refresh(application):
    runtime, github = application
    repo, job = register_and_enqueue(runtime)
    assert Worker(runtime).run_one()
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get(f"/ingestion-jobs/{job}").json()["status"] == "COMPLETE"
        response = client.post(
            f"/repositories/{repo}/query", json={"question": "timeout evidence?"}
        )
        assert response.status_code == 200, response.text
        first_revision = response.json()["revision"]
        assert response.json()["citations"]
    with Session(runtime.engine) as session:
        assert session.scalars(select(Query)).one().query_text is None
        assert session.scalars(select(Answer)).one().answer_text is None
    github.body = "changed timeout behavior"
    _, job2 = register_and_enqueue(runtime, key="second")
    assert Worker(runtime).run_one()
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(
            f"/repositories/{repo}/query",
            json={"question": "changed timeout?", "store_content": True},
        )
        assert response.status_code == 200
        assert response.json()["revision"] != first_revision
        assert any("changed timeout" in s["text"] for s in response.json()["retrieved_sources"])
    with Session(runtime.engine) as session:
        assert len(session.scalars(select(IndexOperation)).all()) == 2
        assert any(q.query_text is not None for q in session.scalars(select(Query)))
        assert session.get(IngestionJob, UUID(job2)).result_revision_id is not None


class Crash(BaseException):
    pass


@pytest.mark.parametrize("stage", ["after_ingestion", "vectors_written", "ready_produced"])
def test_worker_restart_replays_snapshot_and_operation_without_duplicates(application, stage):
    runtime, github = application
    _, identity = register_and_enqueue(runtime)

    def crash(value):
        if value == stage:
            raise Crash()

    with pytest.raises(Crash):
        Worker(runtime, boundary=crash).run_one()
    fetches = github.count
    expire(runtime, identity)
    assert Worker(runtime).run_one()
    assert github.count == fetches
    with Session(runtime.engine) as session:
        job = session.get(IngestionJob, UUID(identity))
        assert job.status == "COMPLETE" and job.attempts == 2
        assert len(session.scalars(select(IndexOperation)).all()) == 1
    assert runtime.store.client.count(runtime.store.collection, exact=True).count == 1
    assert len(list(runtime.settings.application_root.rglob("snapshot.jsonl"))) == 1


@pytest.mark.parametrize("status,retryable", [(503, True), (403, False), (404, False)])
def test_boundary_failures_classified_and_sanitized(application, status, retryable):
    runtime, github = application
    _, identity = register_and_enqueue(runtime)
    github.status = status
    Worker(runtime).run_one()
    with TestClient(create_app(runtime=runtime)) as client:
        body = client.get(f"/ingestion-jobs/{identity}")
        assert body.json()["status"] == "FAILED" and body.json()["retryable"] == retryable
        assert "never-expose" not in body.text
    if retryable:
        github.status = None
        Worker(runtime).run_one()
        with Session(runtime.engine) as session:
            assert session.get(IngestionJob, UUID(identity)).status == "COMPLETE"


def test_query_infrastructure_failure_is_not_successful_empty_answer(application):
    runtime, _ = application
    repo, _ = register_and_enqueue(runtime)
    Worker(runtime).run_one()
    ready = runtime.coordinator().ready("acme/repo")
    runtime.store.delete(ready.ids)
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(f"/repositories/{repo}/query", json={"question": "timeout?"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "retrieval_infrastructure_unavailable"


def test_different_repositories_index_concurrently_without_collection_race(application):
    from concurrent.futures import ThreadPoolExecutor

    runtime, _ = application
    _repo, _ = register_and_enqueue(runtime)
    with TestClient(create_app(runtime=runtime)) as client:
        other = client.post("/repositories", json={"name": "acme/other"}).json()["id"]
        assert (
            client.post(
                f"/repositories/{other}/ingestions", json={"idempotency_key": "other"}
            ).status_code
            == 202
        )
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(lambda _: Worker(runtime).run_one(), range(2)))
    Worker(runtime).run_one()  # A retryable collection initialization race must converge.
    with Session(runtime.engine) as session:
        states = [
            (job.status, job.error_code, job.index_stage)
            for job in session.scalars(select(IngestionJob))
        ]
        assert {state[0] for state in states} == {"COMPLETE"}, states
    assert not set(runtime.coordinator().ready("acme/repo").ids) & set(
        runtime.coordinator().ready("acme/other").ids
    )
    assert runtime.store.client.count(runtime.store.collection, exact=True).count == 2


def test_expired_lease_cannot_publish_ready(application):
    runtime, _ = application
    _, identity = register_and_enqueue(runtime)

    def lose_lease(stage):
        if stage == "verified":
            expire(runtime, identity)

    Worker(runtime, boundary=lose_lease).run_one()
    with Session(runtime.engine) as session:
        assert all(op.status != "READY" for op in session.scalars(select(IndexOperation)))
        assert session.get(IngestionJob, UUID(identity)).status != "COMPLETE"
    Worker(runtime).run_one()
    with Session(runtime.engine) as session:
        assert session.get(IngestionJob, UUID(identity)).status == "COMPLETE"


def test_provider_and_citation_failures_sanitized(application):
    from github_intelligence.generation.provider import LLMProviderError

    runtime, _ = application
    repo, _ = register_and_enqueue(runtime)
    Worker(runtime).run_one()

    class FailingProvider:
        def generate(self, **_):
            raise LLMProviderError("never-echo-provider-secret")

    runtime.provider = FailingProvider()
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(f"/repositories/{repo}/query", json={"question": "private query"})
        assert response.status_code == 502 and "never-echo" not in response.text

    class InvalidCitation:
        def generate(self, **_):
            return DraftAnswer(answer="Unsupported [S999].", citation_ids=["S999"], limitations=[])

    runtime.provider = InvalidCitation()
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(f"/repositories/{repo}/query", json={"question": "private query"})
        assert response.status_code == 502
    with Session(runtime.engine) as session:
        assert all(
            q.status == "failed" and q.query_text is None for q in session.scalars(select(Query))
        )


def test_empty_ready_evidence_is_distinct_from_failure(application):
    runtime, github = application
    # A normal valid scan emits no sources. It creates an honest zero-chunk READY revision.
    original = github.__call__

    def empty(request):
        if request.url.path.endswith("/issues"):
            return httpx.Response(200, json=[])
        return original(request)

    runtime.github_transport = httpx.MockTransport(empty)
    repo, _ = register_and_enqueue(runtime)
    Worker(runtime).run_one()
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.post(f"/repositories/{repo}/query", json={"question": "no evidence"})
        assert response.status_code == 200 and response.json()["status"] == "insufficient_evidence"
        assert response.json()["citations"] == []


def test_health_missing_required_dependencies_returns_503(application):
    runtime, _ = application
    runtime.provider = None
    runtime.settings.openai_api_key = None
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
    runtime.provider = Provider()
    runtime.settings.application_root.rmdir()
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get("/health/ready").status_code == 503


def test_automatic_heartbeat_keeps_slow_worker_owned(application):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    runtime, _ = application
    runtime.settings.worker_lease_seconds = 3
    runtime.settings.worker_heartbeat_seconds = 0.2
    _, identity = register_and_enqueue(runtime)
    entered, release = Event(), Event()

    def slow(stage):
        if stage == "after_ingestion":
            entered.set()
            assert release.wait(15)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(Worker(runtime, boundary=slow).run_one)
        try:
            assert entered.wait(10)
            time.sleep(3.2)
            assert not Worker(runtime).run_one()
            with Session(runtime.engine) as session:
                assert session.get(IngestionJob, UUID(identity)).attempts == 1
        finally:
            release.set()
        assert future.result(timeout=15)


def test_retry_budget_and_rate_limit_schedule_are_bounded(application):
    runtime, github = application
    _, identity = register_and_enqueue(runtime)
    github.status = 503
    for _ in range(runtime.settings.worker_max_attempts):
        assert Worker(runtime).run_one()
    assert not Worker(runtime).run_one()
    with Session(runtime.engine) as session:
        job = session.get(IngestionJob, UUID(identity))
        assert job.status == "FAILED" and not job.retryable
        assert job.attempts == runtime.settings.worker_max_attempts
    github.status = 429

    def rate_limit(request):
        return httpx.Response(429, json={"message": "limited"}, headers={"Retry-After": "120"})

    runtime.github_transport = httpx.MockTransport(rate_limit)
    _, deferred = register_and_enqueue(runtime, key="deferred")
    Worker(runtime).run_one()
    assert not Worker(runtime).run_one()
    with Session(runtime.engine) as session:
        job = session.get(IngestionJob, UUID(deferred))
        assert job.retryable and job.next_attempt_at > session.scalar(
            select(func.now())
        ) + timedelta(seconds=118)


def test_actual_worker_process_exit_and_server_store_recovery(application, tmp_path):
    import subprocess
    import sys

    runtime, _ = application
    _, identity = register_and_enqueue(runtime)

    def crash(stage):
        if stage == "after_ingestion":
            raise Crash()

    with pytest.raises(Crash):
        Worker(runtime, boundary=crash).run_one()
    expire(runtime, identity)
    code = """
import os,sys
from pathlib import Path
from github_intelligence.application.runtime import Runtime
from github_intelligence.application.worker import Worker
from github_intelligence.config import Settings
from github_intelligence.indexing.storage import IncrementalVectorStore
class Encoder:
    dimension=3
    def encode(self,texts): return [[1.0,0.0,0.0] for _ in texts]
settings=Settings(_env_file=None,database_url=os.environ['TEST_DATABASE_URL'],
    application_root=Path(sys.argv[1]),application_qdrant_url=os.environ['TEST_QDRANT_URL'],
    application_embedding_version='test-v1')
store=IncrementalVectorStore(None,sys.argv[2],url=settings.application_qdrant_url)
runtime=Runtime(settings,store=store,encoder=Encoder())
def crash(stage):
    if stage=='vectors_written': os._exit(7)
Worker(runtime,boundary=crash).run_one()
"""
    env = os.environ.copy()
    env["TEST_DATABASE_URL"] = runtime.engine.url.render_as_string(hide_password=False)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(runtime.settings.application_root),
            runtime.store.collection,
        ],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 7, "Child did not reach its crash boundary"
    expire(runtime, identity)
    assert Worker(runtime).run_one()
    with Session(runtime.engine) as session:
        assert session.get(IngestionJob, UUID(identity)).status == "COMPLETE"
        assert len(session.scalars(select(IndexOperation)).all()) == 1
    assert runtime.store.client.count(runtime.store.collection, exact=True).count == 1


def test_separate_api_client_reads_previous_ready_during_worker_publication(application):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    runtime, github = application
    repo, _ = register_and_enqueue(runtime)
    Worker(runtime).run_one()
    previous = runtime.coordinator().ready("acme/repo").revision
    github.body = "changed timeout behavior"
    register_and_enqueue(runtime, key="in-progress")
    api_store = IncrementalVectorStore(
        None, runtime.store.collection, url=runtime.settings.application_qdrant_url
    )
    api_runtime = Runtime(
        runtime.settings,
        engine=runtime.engine,
        store=api_store,
        encoder=Encoder(),
        reranker=Reranker(),
        provider=Provider(),
    )
    entered, release = Event(), Event()

    def pause(stage):
        if stage == "lexical_built":
            entered.set()
            assert release.wait(15)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(Worker(runtime, boundary=pause).run_one)
            try:
                assert entered.wait(10)
                with TestClient(create_app(runtime=api_runtime)) as client:
                    response = client.post(
                        f"/repositories/{repo}/query", json={"question": "timeout?"}
                    )
                    assert response.status_code == 200 and response.json()["revision"] == str(
                        previous
                    )
                    assert all(
                        "changed timeout" not in source["text"]
                        for source in response.json()["retrieved_sources"]
                    )
            finally:
                release.set()
            assert future.result(timeout=15)
        with TestClient(create_app(runtime=api_runtime)) as client:
            response = client.post(
                f"/repositories/{repo}/query", json={"question": "changed timeout?"}
            )
            assert response.status_code == 200 and response.json()["revision"] != str(previous)
    finally:
        api_store.client.close()


def test_duplicate_concurrent_http_creation_is_one_durable_job(application):
    from concurrent.futures import ThreadPoolExecutor

    runtime, _ = application
    with TestClient(create_app(runtime=runtime)) as client:
        repo = client.post("/repositories", json={"name": "acme/repo"}).json()["id"]

        def enqueue(_):
            response = client.post(
                f"/repositories/{repo}/ingestions", json={"idempotency_key": "same"}
            )
            assert response.status_code == 202
            return response.json()["id"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(enqueue, range(2)))
    assert len(set(ids)) == 1
    with Session(runtime.engine) as session:
        assert len(session.scalars(select(IngestionJob)).all()) == 1
