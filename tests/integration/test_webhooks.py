"""Real PostgreSQL receipts/queue and real server Qdrant, canonical GitHub via HTTP transport."""

import base64
import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_application import Encoder, application, expire  # noqa: F401

from github_intelligence.application.api import create_app
from github_intelligence.application.worker import Worker
from github_intelligence.indexing.models import IndexOperation
from github_intelligence.ingestion.snapshot import load_snapshot_report
from github_intelligence.persistence.models import Document, IngestionJob, WebhookDelivery
from github_intelligence.persistence.operations import register_repository

SECRET = "phase-five-webhook-secret-material-32-bytes"


class CountingEncoder(Encoder):
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        return super().encode(texts)


class CanonicalGitHub:
    def __init__(self):
        self.paths = []
        self.body = "Canonical timeout evidence"
        self.comments = True
        self.issue_exists = True
        self.release_exists = True
        self.docs = True
        self.head = "a" * 40
        self.tree = "b" * 40
        self.status = None
        self.native = 42
        self.name = "acme/repo"

    def __call__(self, request):
        path = request.url.path.removeprefix("/repos/acme/repo")
        self.paths.append((path, str(request.url.query)))
        if self.status:
            return httpx.Response(self.status, json={"message": "sensitive-provider-body"})
        root = "https://github.com/acme/repo"
        issue = {
            "id": 1,
            "number": 1,
            "title": "Timeout",
            "body": self.body,
            "html_url": root + "/issues/1",
        }
        pr = {
            "id": 2,
            "number": 2,
            "title": "Timeout PR",
            "body": self.body,
            "html_url": root + "/pull/2",
        }
        comment = {
            "id": 100,
            "body": "Canonical comment evidence",
            "html_url": root + "/issues/1#issuecomment-100",
        }
        inline = {
            "id": 200,
            "body": "Canonical inline evidence",
            "html_url": root + "/pull/2#discussion_r200",
        }
        review = {
            "id": 300,
            "body": "Canonical review evidence",
            "html_url": root + "/pull/2#pullrequestreview-300",
        }
        release = {
            "id": 400,
            "name": "Release",
            "body": self.body,
            "html_url": root + "/releases/tag/v1",
        }
        values = {
            "": {
                "id": self.native,
                "full_name": self.name,
                "html_url": root,
                "default_branch": "main",
                "private": False,
            },
            "/commits/main": {"sha": self.head, "commit": {"tree": {"sha": self.tree}}},
            "/issues": [issue] if self.issue_exists else [],
            "/issues/1": issue,
            "/issues/1/comments": [comment] if self.comments else [],
            "/pulls": [pr],
            "/pulls/2": pr,
            "/issues/2/comments": [comment] if self.comments else [],
            "/pulls/2/comments": [inline] if self.comments else [],
            "/pulls/2/reviews": [review],
            "/releases": [release] if self.release_exists else [],
            "/releases/400": release,
            "/commits": [
                {
                    "sha": self.head,
                    "html_url": root + "/commit/" + self.head,
                    "commit": {"message": "Immutable commit"},
                }
            ],
            "/git/trees/" + self.tree: {
                "sha": self.tree,
                "truncated": False,
                "tree": [{"type": "blob", "path": "README.md", "sha": "c" * 40}]
                if self.docs
                else [],
            },
            "/git/blobs/" + "c" * 40: {
                "sha": "c" * 40,
                "encoding": "base64",
                "content": base64.b64encode(b"Documentation timeout evidence").decode(),
            },
        }
        if (path == "/issues/1" and not self.issue_exists) or (
            path == "/releases/400" and not self.release_exists
        ):
            return httpx.Response(404, json={"message": "Not found"})
        if path not in values:
            return httpx.Response(404, json={"message": "Unknown test endpoint"})
        return httpx.Response(200, json=values[path])


@pytest.fixture
def webhook_application(application):  # noqa: F811
    runtime, _ = application
    runtime.settings.github_webhooks_enabled = True
    runtime.settings.github_webhook_secret = SECRET
    runtime.settings.github_webhook_max_body_bytes = 1024
    boundary = CanonicalGitHub()
    runtime.github_transport = httpx.MockTransport(boundary)
    runtime.encoder = CountingEncoder()
    with Session(runtime.engine, expire_on_commit=False) as session, session.begin():
        repo = register_repository(session, "acme/repo", github_id="42", default_branch="main")
        repo_id = repo.id
    return runtime, boundary, repo_id


def payload(event="issues", action="edited"):
    body = {
        "repository": {"id": 42, "full_name": "acme/repo", "default_branch": "main"},
        "action": action,
    }
    if event in {"issues", "issue_comment"}:
        body["issue"] = {"id": 1, "number": 1}
    elif event in {"pull_request", "pull_request_review", "pull_request_review_comment"}:
        body["pull_request"] = {"id": 2, "number": 2}
    if event == "issue_comment":
        body["comment"] = {"id": 100}
    if event == "pull_request_review_comment":
        body["comment"] = {"id": 200}
    if event == "pull_request_review":
        body["review"] = {"id": 300}
    if event == "release":
        body["release"] = {"id": 400}
    if event == "push":
        body.update(ref="refs/heads/main", after="d" * 40, forced=False, deleted=False)
    return body


def send(runtime, *, event="issues", body=None, delivery_id=None, raw=None, signature=None):
    raw = (
        raw
        if raw is not None
        else json.dumps(body if body is not None else payload(event)).encode()
    )
    headers = {
        "x-github-delivery": str(delivery_id or uuid4()),
        "x-github-event": event,
        "x-hub-signature-256": signature
        or "sha256=" + hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest(),
    }
    with TestClient(create_app(runtime=runtime)) as client:
        return client.post("/webhooks/github", content=raw, headers=headers)


def job(runtime, response):
    with Session(runtime.engine) as session:
        return session.get(IngestionJob, UUID(response.json()["job_id"]))


def ready(runtime, repo_id):
    return runtime.coordinator().ready("acme/repo")


@pytest.mark.parametrize(
    "event,expected",
    [
        ("issues", "/issues/1"),
        ("issue_comment", "/issues/1"),
        ("pull_request", "/pulls/2"),
        ("pull_request_review", "/pulls/2"),
        ("pull_request_review_comment", "/pulls/2"),
        ("release", "/releases/400"),
        ("push", "/commits/main"),
    ],
)
def test_supported_events_refetch_only_canonical_scope(webhook_application, event, expected):
    runtime, boundary, repo_id = webhook_application
    response = send(runtime, event=event)
    assert response.status_code == 202 and boundary.paths == []
    assert Worker(runtime).run_one()
    result = job(runtime, response)
    assert result.status == "COMPLETE", result.error_code
    assert expected in [p[0] for p in boundary.paths]
    report = load_snapshot_report(Path(result.snapshot_path))
    assert report.complete_for_requested_scope
    if event not in {"push"}:
        assert "/commits/main" not in [p[0] for p in boundary.paths]
        assert "/issues" not in [p[0] for p in boundary.paths]
        assert "/pulls" not in [p[0] for p in boundary.paths]
    else:
        assert "/git/trees/" + boundary.tree in [p[0] for p in boundary.paths]
    assert ready(runtime, repo_id).revision == result.result_revision_id
    with Session(runtime.engine) as session:
        assert session.get(WebhookDelivery, response.json()["delivery_id"]).status == "COMPLETE"


@pytest.mark.parametrize("signature", ["sha256=" + "0" * 64, "sha1=abc", "garbage"])
def test_invalid_signature_does_not_parse_or_persist(webhook_application, signature, monkeypatch):
    runtime, _, _ = webhook_application
    from github_intelligence.application import webhooks

    def forbidden(*args, **kwargs):
        pytest.fail("Parsed before signature verification")

    monkeypatch.setattr(webhooks.json, "loads", forbidden)
    response = send(runtime, raw=b"not-json", signature=signature)
    assert response.status_code == 401
    with Session(runtime.engine) as session:
        assert session.scalars(select(WebhookDelivery)).all() == []


def test_missing_signature_tampering_body_limit_and_disabled(webhook_application):
    runtime, _, _ = webhook_application
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.post("/webhooks/github", content=b"{}").status_code == 401
        raw = json.dumps(payload()).encode()
        signature = "sha256=" + hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()
        assert (
            client.post(
                "/webhooks/github", content=raw + b" ", headers={"x-hub-signature-256": signature}
            ).status_code
            == 401
        )
    assert send(runtime, raw=b"x" * 1025).status_code == 413
    runtime.settings.github_webhooks_enabled = False
    assert send(runtime).status_code == 503


def test_duplicates_concurrent_receipts_and_restart(webhook_application):
    runtime, boundary, _ = webhook_application
    delivery_id = uuid4()
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: send(runtime, delivery_id=delivery_id), range(4)))
    assert sorted(r.status_code for r in responses) == [200, 200, 200, 202]
    assert len({r.json()["job_id"] for r in responses}) == 1
    assert boundary.paths == []
    assert send(runtime, delivery_id=delivery_id).json()["duplicate"]
    with Session(runtime.engine) as session:
        assert len(session.scalars(select(WebhookDelivery)).all()) == 1
        assert len(session.scalars(select(IngestionJob)).all()) == 1
    assert Worker(runtime).run_one()
    assert send(runtime, delivery_id=delivery_id).json()["status"] == "COMPLETE"
    assert send(runtime, delivery_id=delivery_id, body=payload(action="closed")).status_code == 409


@pytest.mark.parametrize(
    "event,body,code,status",
    [
        ("ping", {}, "unsupported_event", 200),
        ("issues", {}, "malformed_payload", 400),
        (
            "issues",
            {"action": "edited", "repository": {"id": 999, "full_name": "unknown/repo"}},
            "repository_not_registered",
            200,
        ),
    ],
)
def test_terminal_receipts_are_durable_without_work(webhook_application, event, body, code, status):
    runtime, _, _ = webhook_application
    response = send(runtime, event=event, body=body)
    assert response.status_code == status and response.json()["code"] == code
    with Session(runtime.engine) as session:
        assert session.get(WebhookDelivery, response.json()["delivery_id"]) is not None
        assert session.scalars(select(IngestionJob)).all() == []


def test_raw_content_secrets_are_not_retained(webhook_application, caplog):
    runtime, _, _ = webhook_application
    body = payload()
    body.update(Authorization="private-token", github_token="private-token", secret=SECRET)
    response = send(runtime, body=body)
    with Session(runtime.engine) as session:
        delivery = session.get(WebhookDelivery, response.json()["delivery_id"])
        assert "private-token" not in str(delivery.event_metadata)
        assert SECRET not in str(delivery.event_metadata)
        assert not hasattr(delivery, "raw_body")
        assert "private-token" not in str(session.get(IngestionJob, delivery.job_id).scope)
    assert SECRET not in caplog.text and "private-token" not in caplog.text


def test_rename_and_native_identity_conflict_are_deferred(webhook_application):
    runtime, _, repo_id = webhook_application
    body = payload()
    body["repository"]["full_name"] = "new/repo"
    response = send(runtime, body=body)
    assert response.json()["code"] == "repository_rename_requires_operator"
    body = payload()
    body["repository"]["id"] = 43
    assert send(runtime, body=body).json()["code"] == "repository_identity_conflict"
    with TestClient(create_app(runtime=runtime)) as client:
        assert client.get(f"/repositories/{repo_id}").json()["github_id"] == "42"


def test_non_default_push_is_ignored_force_and_late_push_use_current_head(webhook_application):
    runtime, boundary, repo_id = webhook_application
    body = payload("push")
    body["ref"] = "refs/heads/feature"
    assert send(runtime, event="push", body=body).json()["code"] == "branch_not_indexed"
    body = payload("push")
    body["forced"] = True
    body["after"] = "1" * 40
    boundary.head = "e" * 40
    boundary.tree = "f" * 40
    first = send(runtime, event="push", body=body)
    assert Worker(runtime).run_one() and job(runtime, first).status == "COMPLETE"
    for path, query in boundary.paths:
        if path == "/commits":
            assert boundary.head in query
    boundary.paths.clear()
    boundary.docs = False
    boundary.tree = "9" * 40
    boundary.head = "8" * 40
    second = send(runtime, event="push", body=body)
    assert Worker(runtime).run_one() and job(runtime, second).status == "COMPLETE"
    with Session(runtime.engine) as session:
        docs = session.scalars(
            select(Document).where(Document.source_type == "documentation")
        ).all()
        assert len(docs) == 1 and docs[0].deleted_at is not None
    assert ready(runtime, repo_id).revision == job(runtime, second).result_revision_id


def test_pending_events_coalesce_and_unchanged_events_do_not_embed(webhook_application):
    runtime, _, repo_id = webhook_application
    first = send(runtime)
    second = send(runtime, event="issue_comment")
    assert first.json()["job_id"] == second.json()["job_id"]
    assert Worker(runtime).run_one()
    runtime.encoder.calls.clear()
    delayed_body = payload(action="closed")
    delayed_body["issue"].update(body="Old payload content", updated_at="2020-01-01T00:00:00Z")
    third = send(runtime, body=delayed_body)  # Older event still fetches current state.
    assert third.json()["job_id"] != first.json()["job_id"]
    assert Worker(runtime).run_one() and job(runtime, third).status == "COMPLETE"
    assert runtime.encoder.calls == []
    with Session(runtime.engine) as session:
        delivery = session.get(WebhookDelivery, third.json()["delivery_id"])
        assert delivery.event_metadata["source_updated_at"] == "2020-01-01T00:00:00+00:00"
        issue = session.scalars(select(Document).where(Document.source_type == "issue")).one()
        assert "Old payload content" not in issue.normalized_payload["text"]
    assert ready(runtime, repo_id).revision == job(runtime, third).result_revision_id


@pytest.mark.parametrize(
    "event", ["issues", "issue_comment", "release", "pull_request_review_comment"]
)
def test_explicit_deletion_is_narrow_and_preserves_previous_ready(webhook_application, event):
    runtime, boundary, repo_id = webhook_application
    send(runtime, event=event)
    Worker(runtime).run_one()
    old = ready(runtime, repo_id)
    if event == "issues":
        boundary.issue_exists = False
    elif event == "release":
        boundary.release_exists = False
    else:
        boundary.comments = False
    deletion = send(runtime, event=event, body=payload(event, "deleted"))
    Worker(runtime).run_one()
    assert job(runtime, deletion).status == "COMPLETE"
    with Session(runtime.engine) as session:
        operation = session.get(IndexOperation, job(runtime, deletion).result_revision_id)
        assert operation.plan["counts"]["deleted"] == 1
    # Old vectors are retained for the previous revision's readers.
    assert old.vector_search([1.0, 0.0, 0.0], top_k=10)


def test_absence_and_delayed_delete_cannot_remove_current_content(webhook_application):
    runtime, boundary, repo_id = webhook_application
    send(runtime)
    Worker(runtime).run_one()
    boundary.comments = False
    send(runtime, event="issue_comment")
    Worker(runtime).run_one()
    with Session(runtime.engine) as session:
        comment = session.scalars(
            select(Document).where(Document.source_type == "issue_comment")
        ).one()
        assert comment.deleted_at is None
    boundary.comments = True
    delayed = send(runtime, event="issue_comment", body=payload("issue_comment", "deleted"))
    Worker(runtime).run_one()
    assert job(runtime, delayed).status == "COMPLETE"
    with Session(runtime.engine) as session:
        assert (
            session.get(IndexOperation, job(runtime, delayed).result_revision_id).plan["counts"][
                "deleted"
            ]
            == 0
        )
    assert ready(runtime, repo_id)


@pytest.mark.parametrize("stage", ["after_ingestion", "vectors_written", "ready_produced"])
def test_worker_crash_replays_delivery_job_and_snapshot(webhook_application, stage):
    runtime, boundary, _ = webhook_application
    response = send(runtime)

    class Crash(BaseException):
        pass

    def interrupt(value):
        if value == stage:
            raise Crash()

    with pytest.raises(Crash):
        Worker(runtime, boundary=interrupt).run_one()
    requests = len(boundary.paths)
    expire(runtime, response.json()["job_id"])
    assert Worker(runtime).run_one()
    assert job(runtime, response).status == "COMPLETE"
    assert len(boundary.paths) == requests
    with Session(runtime.engine) as session:
        assert len(session.scalars(select(IndexOperation)).all()) == 1
        assert session.get(WebhookDelivery, response.json()["delivery_id"]).status == "COMPLETE"


def test_active_job_is_not_coalesced_previous_ready_visible_until_verified(webhook_application):
    runtime, boundary, repo_id = webhook_application
    send(runtime)
    Worker(runtime).run_one()
    old = ready(runtime, repo_id)
    boundary.body = "Changed canonical evidence"
    second = send(runtime)
    reached, release = Event(), Event()

    def pause(stage):
        if stage == "lexical_built":
            reached.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(Worker(runtime, boundary=pause).run_one)
        assert reached.wait(10)
        third = send(runtime, event="issue_comment")
        assert third.json()["job_id"] != second.json()["job_id"]
        assert Worker(runtime).run_one() is False  # Same repository waits for predecessor.
        assert ready(runtime, repo_id).revision == old.revision
        release.set()
        assert future.result(timeout=10)
    assert ready(runtime, repo_id).revision == job(runtime, second).result_revision_id
    assert Worker(runtime).run_one() and job(runtime, third).status == "COMPLETE"


def test_transient_failure_is_durable_and_authorization_is_terminal(webhook_application):
    runtime, boundary, _ = webhook_application
    response = send(runtime)
    boundary.status = 503
    Worker(runtime).run_one()
    failed = job(runtime, response)
    assert failed.status == "FAILED" and failed.retryable
    assert "sensitive" not in failed.error_detail
    boundary.status = None
    Worker(runtime).run_one()
    assert job(runtime, response).status == "COMPLETE"
    boundary.status = 403
    response = send(runtime)
    Worker(runtime).run_one()
    assert job(runtime, response).status == "FAILED" and not job(runtime, response).retryable


def test_authenticated_malformed_json_and_malformed_headers(webhook_application):
    runtime, _, _ = webhook_application
    response = send(runtime, raw=b"{bad-json")
    assert response.status_code == 400 and response.json()["code"] == "malformed_payload"
    with Session(runtime.engine) as session:
        assert session.get(WebhookDelivery, response.json()["delivery_id"]).status == "FAILED"
    with TestClient(create_app(runtime=runtime)) as client:
        raw = json.dumps(payload()).encode()
        signature = "sha256=" + hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()
        assert (
            client.post(
                "/webhooks/github", content=raw, headers={"x-hub-signature-256": signature}
            ).status_code
            == 400
        )


def test_new_registered_native_identity_is_verified_during_refetch(webhook_application):
    runtime, boundary, repo_id = webhook_application
    from github_intelligence.persistence.models import Repository

    with Session(runtime.engine) as session, session.begin():
        session.get(Repository, repo_id).github_id = None
    response = send(runtime)
    boundary.native = 43
    Worker(runtime).run_one()
    assert job(runtime, response).status == "FAILED" and not job(runtime, response).retryable
    with Session(runtime.engine) as session:
        assert session.get(Repository, repo_id).github_id is None
        assert session.scalars(select(IndexOperation)).all() == []


def test_comment_on_pull_request_refreshes_full_pr_conversation(webhook_application):
    runtime, boundary, _ = webhook_application
    body = payload("issue_comment")
    body["issue"] = {"id": 2, "number": 2, "pull_request": {}}
    response = send(runtime, event="issue_comment", body=body)
    Worker(runtime).run_one()
    assert job(runtime, response).status == "COMPLETE"
    assert "/pulls/2" in [p[0] for p in boundary.paths]
    assert "/pulls/2/reviews" in [p[0] for p in boundary.paths]
    with Session(runtime.engine) as session:
        assert {d.source_type for d in session.scalars(select(Document))} == {
            "pull_request",
            "pull_request_comment",
            "pull_request_review_comment",
            "pull_request_review",
        }


def test_ambiguous_not_found_does_not_authorize_deletion(webhook_application):
    runtime, boundary, repo_id = webhook_application
    send(runtime)
    Worker(runtime).run_one()
    old = ready(runtime, repo_id)
    boundary.issue_exists = False
    response = send(runtime)  # No signed deletion event.
    Worker(runtime).run_one()
    assert job(runtime, response).status == "FAILED"
    assert ready(runtime, repo_id).revision == old.revision
    with Session(runtime.engine) as session:
        assert all(d.deleted_at is None for d in session.scalars(select(Document)))


def test_pending_delete_can_widen_missing_permission_but_active_scope_is_immutable(
    webhook_application,
):
    runtime, boundary, _ = webhook_application
    send(runtime)
    Worker(runtime).run_one()
    update = send(runtime)
    deletion = send(runtime, body=payload("issues", "deleted"))
    assert update.json()["job_id"] == deletion.json()["job_id"]
    assert job(runtime, update).scope["options"]["allow_missing"] is True
    boundary.issue_exists = False
    Worker(runtime).run_one()
    assert job(runtime, update).status == "COMPLETE"
    with Session(runtime.engine) as session:
        issue = session.scalars(select(Document).where(Document.source_type == "issue")).one()
        assert issue.deleted_at is not None


def test_manual_reconciliation_remains_available_after_terminal_fetch_failure(webhook_application):
    runtime, boundary, repo_id = webhook_application
    response = send(runtime)
    boundary.status = 403
    Worker(runtime).run_one()
    assert job(runtime, response).status == "FAILED"
    boundary.status = None
    with TestClient(create_app(runtime=runtime)) as client:
        manual = client.post(
            f"/repositories/{repo_id}/ingestions",
            json={"idempotency_key": "reconcile-after-downtime", "reconcile": True},
        )
        assert manual.status_code == 202
    assert Worker(runtime).run_one()
    with Session(runtime.engine) as session:
        assert session.get(IngestionJob, UUID(manual.json()["id"])).status == "COMPLETE"
    assert "/issues" in [p[0] for p in boundary.paths]


def test_delivery_status_and_webhook_schema_readiness(webhook_application):
    runtime, _, _ = webhook_application
    from sqlalchemy import text

    response = send(runtime)
    with TestClient(create_app(runtime=runtime)) as client:
        status = client.get("/webhook-deliveries/" + response.json()["delivery_id"])
        assert status.status_code == 200 and status.json()["job"]["status"] == "QUEUED"
        assert client.get("/health/ready").status_code == 200
        with runtime.engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE webhook_deliveries RENAME TO webhook_deliveries_hidden")
            )
        assert client.get("/health/ready").status_code == 503
        assert send(runtime).status_code == 503
        with runtime.engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE webhook_deliveries_hidden RENAME TO webhook_deliveries")
            )


def test_explicit_operator_retry_replays_failed_index_without_new_operation(webhook_application):
    runtime, boundary, repo_id = webhook_application
    runtime.settings.worker_max_attempts = 1
    response = send(runtime)

    def fail(stage):
        if stage == "vectors_written":
            raise ValueError("operator-must-repair-this-invariant")

    Worker(runtime, boundary=fail).run_one()
    failed = job(runtime, response)
    assert failed.status == "FAILED" and not failed.retryable
    assert failed.attempts == 1
    requests = len(boundary.paths)
    with TestClient(create_app(runtime=runtime)) as client:
        retried = client.post(f"/ingestion-jobs/{failed.id}/retry")
        assert retried.status_code == 202
        assert retried.json()["attempts"] == 1
        assert client.post(f"/ingestion-jobs/{failed.id}/retry").status_code == 409
        assert client.post(f"/ingestion-jobs/{uuid4()}/retry").status_code == 404
    Worker(runtime).run_one()
    assert job(runtime, response).status == "COMPLETE" and job(runtime, response).attempts == 2
    assert requests == len(boundary.paths)
    with Session(runtime.engine) as session:
        assert len(session.scalars(select(IndexOperation)).all()) == 1
    assert ready(runtime, repo_id).revision == job(runtime, response).result_revision_id


def test_limited_target_scan_cannot_remove_or_publish_partial_state(webhook_application):
    runtime, boundary, repo_id = webhook_application
    send(runtime)
    Worker(runtime).run_one()
    previous = ready(runtime, repo_id)
    runtime.settings.github_max_pages = 1
    canonical = runtime.github_transport

    def limited(request):
        if request.url.path.endswith("/issues/1/comments"):
            return httpx.Response(
                200,
                json=[],
                headers={
                    "link": (
                        "<https://api.github.com/repos/acme/repo/issues/1/comments?page=2>"
                        '; rel="next"'
                    )
                },
            )
        return boundary(request)

    runtime.github_transport = httpx.MockTransport(limited)
    response = send(runtime, event="issue_comment", body=payload("issue_comment", "deleted"))
    Worker(runtime).run_one()
    assert job(runtime, response).status == "FAILED"
    assert ready(runtime, repo_id).revision == previous.revision
    with Session(runtime.engine) as session:
        assert all(document.deleted_at is None for document in session.scalars(select(Document)))
    runtime.github_transport = canonical


def test_busy_database_is_not_acknowledged_before_durable_receipt(webhook_application):
    runtime, _, repo_id = webhook_application
    from github_intelligence.persistence.models import Repository

    identity = uuid4()
    with Session(runtime.engine) as session, session.begin():
        session.execute(select(Repository).where(Repository.id == repo_id).with_for_update())
        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(send, runtime, delivery_id=identity).result(timeout=5)
        assert response.status_code == 503
    with Session(runtime.engine) as session:
        assert session.get(WebhookDelivery, str(identity)) is None
    assert send(runtime, delivery_id=identity).status_code == 202


def test_repository_rename_event_preserves_native_identity_and_defers(webhook_application):
    runtime, _, repo_id = webhook_application
    response = send(
        runtime,
        event="repository",
        body={"action": "renamed", "repository": {"id": 42, "full_name": "acme/renamed"}},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "DEFERRED"
    assert response.json()["code"] == "repository_rename_requires_operator"
    with Session(runtime.engine) as session:
        delivery = session.get(WebhookDelivery, response.json()["delivery_id"])
        assert delivery.repository_id == repo_id and delivery.repository_github_id == "42"
        assert delivery.job_id is None
        from github_intelligence.persistence.models import Repository

        assert len(session.scalars(select(Repository)).all()) == 1


@pytest.mark.parametrize("name", [None, 123])
def test_malformed_repository_name_is_a_durable_terminal_receipt(webhook_application, name):
    runtime, _, _ = webhook_application
    body = payload()
    body["repository"]["full_name"] = name
    response = send(runtime, body=body)
    assert response.status_code == 400 and response.json()["status"] == "FAILED"
    with Session(runtime.engine) as session:
        assert (
            session.get(WebhookDelivery, response.json()["delivery_id"]).error_code
            == "malformed_payload"
        )
        assert session.scalars(select(IngestionJob)).all() == []
