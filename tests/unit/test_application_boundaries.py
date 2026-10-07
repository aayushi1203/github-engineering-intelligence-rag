import pytest

pytest.importorskip("sqlalchemy")

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from github_intelligence.application.failures import classify
from github_intelligence.application.runtime import Runtime
from github_intelligence.application.worker import Worker
from github_intelligence.config import Settings
from github_intelligence.indexing.storage import IncrementalVectorStore


@pytest.mark.parametrize(
    "failure,retryable",
    [
        (ValueError("never-echo-configuration"), False),
        (PermissionError("never-echo-permission"), False),
        (FileNotFoundError("never-echo-snapshot"), False),
        (ConnectionError("never-echo-network"), True),
        (ResponseHandlingException(ValueError("never-echo-malformed")), False),
        (ResponseHandlingException(httpx.ConnectError("never-echo-transport")), True),
        (UnexpectedResponse(503, "error", b"never-echo-response", httpx.Headers()), True),
        (UnexpectedResponse(403, "error", b"never-echo-response", httpx.Headers()), False),
    ],
)
def test_failures_classified_without_sensitive_details(failure, retryable):
    result = classify(failure)
    assert result.retryable == retryable
    assert "never-echo" not in repr(result)


def test_application_requires_explicit_version_server_and_valid_lease_settings():
    with pytest.raises(ValueError, match="VERSION"):
        Runtime(Settings(_env_file=None, application_embedding_version=None))
    with pytest.raises(ValueError, match="server-mode"):
        Runtime(
            Settings(
                _env_file=None,
                application_embedding_version="pinned",
                application_qdrant_mode="embedded",
            )
        )
    with pytest.raises(ValueError, match="Heartbeat"):
        Runtime(
            Settings(
                _env_file=None,
                application_embedding_version="pinned",
                worker_lease_seconds=3,
                worker_heartbeat_seconds=2,
            )
        )


def test_qdrant_server_url_rejects_credentials_before_constructing_client():
    with pytest.raises(ValueError, match="credentials"):
        IncrementalVectorStore(None, "test", url="https://user:never-echo-password@example.com")


def test_idle_worker_exits_cleanly_on_shutdown():
    from types import SimpleNamespace

    runtime = SimpleNamespace(
        settings=SimpleNamespace(worker_concurrency=1, worker_poll_seconds=0.1)
    )
    worker = Worker(runtime)
    worker.stop.set()
    worker.run()
