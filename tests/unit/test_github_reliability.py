from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from github_intelligence.config import Settings
from github_intelligence.github.client import (
    GitHubClient,
    GitHubError,
    RateLimitError,
    TransientGitHubError,
)
from github_intelligence.github.scans import PaginationCheckpoint, ScanStatus


def settings(path: Path, **updates) -> Settings:
    return Settings(_env_file=None, github_cache_dir=path, github_token=None, **updates)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header, expected", [("7", 7), (format_datetime(datetime.fromtimestamp(1007, UTC)), 7)]
)
async def test_retry_after_delta_and_http_date(
    tmp_path: Path, header: str, expected: float
) -> None:
    calls, waits = [], []

    async def sleep(delay):
        waits.append(delay)

    def handler(request):
        calls.append(request)
        return httpx.Response(
            429 if len(calls) == 1 else 200, headers={"Retry-After": header}, json={"ok": True}
        )

    async with GitHubClient(
        settings(tmp_path), httpx.MockTransport(handler), sleep=sleep, clock=lambda: 1000
    ) as client:
        assert await client.get_json("/items", use_cache=False) == {"ok": True}
    assert waits == [expected] and len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    ["You have exceeded a secondary rate limit", "You triggered an abuse detection mechanism"],
)
async def test_secondary_rate_limits_have_exponential_wait_and_bounded_typed_exhaustion(
    tmp_path: Path, message: str
) -> None:
    calls, waits = [], []

    async def sleep(delay):
        waits.append(delay)

    def handler(request):
        calls.append(request)
        return httpx.Response(
            403, headers={"x-ratelimit-remaining": "42"}, json={"message": message}
        )

    async with GitHubClient(
        settings(tmp_path, github_max_retries=3), httpx.MockTransport(handler), sleep=sleep
    ) as client:
        with pytest.raises(RateLimitError) as caught:
            await client.get_json("/items")
    assert waits == [60, 120] and len(calls) == 3
    assert caught.value.attempts == 3 and caught.value.exhausted
    assert caught.value.kind == "secondary_rate_limit"


@pytest.mark.asyncio
async def test_primary_reset_and_retry_after_are_both_respected(tmp_path: Path) -> None:
    waits = []
    calls = 0

    async def sleep(delay):
        waits.append(delay)

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            403 if calls == 1 else 200,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1010", "retry-after": "3"},
            json=[],
        )

    async with GitHubClient(
        settings(tmp_path), httpx.MockTransport(handler), sleep=sleep, clock=lambda: 1000
    ) as client:
        await client.get_json("/items")
    assert waits == [10]


@pytest.mark.asyncio
async def test_long_retry_after_defers_without_early_retry(tmp_path: Path) -> None:
    waits = []

    async def sleep(delay):
        waits.append(delay)

    async with GitHubClient(
        settings(tmp_path, github_max_retry_wait_seconds=5),
        httpx.MockTransport(lambda req: httpx.Response(429, headers={"Retry-After": "20"})),
        sleep=sleep,
    ) as client:
        with pytest.raises(RateLimitError) as caught:
            await client.get_json("/items")
    assert waits == [] and caught.value.deferred and caught.value.retry_after == 20
    assert caught.value.attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["503", "transport"])
async def test_transient_exhaustion_is_typed_and_sanitized(tmp_path: Path, mode: str) -> None:
    waits = []

    async def sleep(delay):
        waits.append(delay)

    def handler(request):
        if mode == "transport":
            raise httpx.ReadTimeout("dummy-secret-response", request=request)
        return httpx.Response(503, json={"message": "dummy-secret-response"})

    async with GitHubClient(
        settings(tmp_path, github_max_retries=3), httpx.MockTransport(handler), sleep=sleep
    ) as client:
        with pytest.raises(TransientGitHubError) as caught:
            await client.get_json("/items")
    assert len(waits) == 2 and waits[0] >= 0.5 and waits[1] >= 1
    assert caught.value.attempts == 3 and caught.value.exhausted
    assert "dummy-secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_non_rate_limit_403_and_invalid_json_are_not_retried(tmp_path: Path) -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"message": "permission denied; dummy-secret"})

    async with GitHubClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubError) as caught:
            await client.get_json("/items")
    assert len(calls) == 1 and caught.value.kind == "request_rejected"
    async with GitHubClient(
        settings(tmp_path), httpx.MockTransport(lambda req: httpx.Response(200, text="not JSON"))
    ) as client:
        with pytest.raises(GitHubError, match="invalid_json_response"):
            await client.get_json("/invalid")


@pytest.mark.asyncio
async def test_invalid_retry_headers_fall_back_without_nan_or_infinite_wait(tmp_path: Path) -> None:
    waits = []
    calls = 0

    async def sleep(delay):
        waits.append(delay)

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429 if calls == 1 else 200, headers={"retry-after": "NaN"}, json=[])

    async with GitHubClient(
        settings(tmp_path), httpx.MockTransport(handler), sleep=sleep
    ) as client:
        await client.get_json("/items")
    assert waits == [60]


@pytest.mark.asyncio
async def test_cache_is_credential_and_api_isolated_and_fresh_revalidates(tmp_path: Path) -> None:
    calls = []
    current = [1]

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"value": current[0]}, headers={"etag": '"revision"'})

    first_settings = settings(tmp_path, github_api_url="https://api.github.com")
    first_settings.github_token = "test-credential-A"
    async with GitHubClient(first_settings, httpx.MockTransport(handler)) as first:
        assert (await first.get_json("/items"))["value"] == 1
        current[0] = 2
        assert (await first.get_json("/items"))["value"] == 1
        assert (await first.get_json("/items", fresh=True))["value"] == 2
        assert calls[-1].headers["if-none-match"] == '"revision"'
    other_settings = settings(tmp_path)
    other_settings.github_token = "test-credential-B"
    async with GitHubClient(other_settings, httpx.MockTransport(handler)) as other:
        assert (await other.get_json("/items"))["value"] == 2
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(handler)) as anonymous:
        await anonymous.get_json("/items")
    assert len(calls) == 4
    for path in tmp_path.rglob("*"):
        assert "test-credential" not in str(path)
        if path.is_file():
            assert "test-credential" not in path.read_text()


@pytest.mark.asyncio
async def test_fresh_304_full_page_discovers_new_navigation(tmp_path: Path) -> None:
    requests = []

    def handler(request):
        requests.append(request)
        page = request.url.params.get("page")
        if request.headers.get("if-none-match"):
            return httpx.Response(304)  # Omitted navigation must not reuse old end-of-list Link.
        if page == "1":
            return httpx.Response(
                200,
                json=[{"id": 1}],
                headers={
                    "etag": '"first"',
                    "link": '<https://api.github.com/items?page=1>; rel="prev"',
                },
            )
        return httpx.Response(200, json=[])

    async with GitHubClient(
        settings(tmp_path, github_per_page=1), httpx.MockTransport(handler)
    ) as client:
        assert [item async for item in client.paginate("/items")] == [{"id": 1}]
        cp = PaginationCheckpoint()
        assert [item async for item in client.paginate("/items", fresh=True, checkpoint=cp)] == [
            {"id": 1}
        ]
    assert [r.url.params["page"] for r in requests] == ["1", "1", "2"]
    assert cp.status == ScanStatus.COMPLETE and cp.pages_fetched == 2


@pytest.mark.asyncio
async def test_link_pagination_and_max_page_checkpoint(tmp_path: Path) -> None:
    def handler(request):
        page = request.url.params["page"]
        return httpx.Response(
            200,
            json=[{"id": int(page), "updated_at": f"2026-01-0{page}T00:00:00Z"}],
            headers={"link": f'<https://api.github.com/items?page={int(page) + 1}>; rel="next"'}
            if page == "1"
            else {"link": '<https://api.github.com/items?page=1>; rel="prev"'},
        )

    async with GitHubClient(
        settings(tmp_path, github_per_page=100), httpx.MockTransport(handler)
    ) as client:
        cp = PaginationCheckpoint()
        assert len([item async for item in client.paginate("/items", checkpoint=cp)]) == 2
        assert cp.status == ScanStatus.COMPLETE and cp.next_page is None
        assert cp.last_source_id == "2" and cp.max_source_updated_at == datetime(
            2026, 1, 2, tzinfo=UTC
        )
    async with GitHubClient(
        settings(tmp_path, github_max_pages=1), httpx.MockTransport(handler)
    ) as client:
        cp = PaginationCheckpoint()
        assert len([item async for item in client.paginate("/items", checkpoint=cp)]) == 1
        assert cp.status == ScanStatus.LIMITED and cp.reason == "max_pages"
        assert cp.next_page == 2 and cp.last_page == 1


@pytest.mark.asyncio
async def test_malformed_page_and_cross_origin_links_fail_without_skipping_items(
    tmp_path: Path,
) -> None:
    async with GitHubClient(
        settings(tmp_path),
        httpx.MockTransport(lambda req: httpx.Response(200, json=[{"id": 1}, "bad-item"])),
    ) as client:
        cp = PaginationCheckpoint()
        with pytest.raises(GitHubError):
            _ = [item async for item in client.paginate("/items", checkpoint=cp)]
        assert cp.status == ScanStatus.FAILED and cp.items_seen == 0
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200, json=[], headers={"link": '<https://untrusted.example/items?page=2>; rel="next"'}
        )

    async with GitHubClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        with pytest.raises(GitHubError, match="cross_origin"):
            _ = [item async for item in client.paginate("/items", fresh=True)]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_safe_redirects_preserve_repository_identity_without_forwarding_credentials(
    tmp_path: Path,
) -> None:
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/renamed"})
        if request.url.path == "/external":
            return httpx.Response(301, headers={"location": "https://untrusted.example/data"})
        return httpx.Response(200, json={"id": 42})

    async with GitHubClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        assert await client.get_json("/old") == {"id": 42}
        with pytest.raises(GitHubError, match="cross_origin"):
            await client.get_json("/external")
    assert [r.url.path for r in calls] == ["/old", "/renamed", "/external"]


@pytest.mark.asyncio
async def test_page_two_failure_preserves_checkpoint_without_marking_complete(
    tmp_path: Path,
) -> None:
    def handler(request):
        if request.url.params["page"] == "1":
            return httpx.Response(
                200,
                json=[{"id": 9}],
                headers={"link": '<https://api.github.com/items?page=2>; rel="next"'},
            )
        return httpx.Response(500)

    async with GitHubClient(
        settings(tmp_path, github_max_retries=1), httpx.MockTransport(handler)
    ) as client:
        cp = PaginationCheckpoint()
        with pytest.raises(TransientGitHubError):
            _ = [item async for item in client.paginate("/items", checkpoint=cp)]
    assert cp.status == ScanStatus.FAILED and cp.reason == "transient_http_failure"
    assert cp.pages_fetched == 1 and cp.items_seen == 1 and cp.next_page == 2


@pytest.mark.asyncio
async def test_use_cache_false_never_reads_or_writes_cache(tmp_path: Path) -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"number": len(requests)})

    async with GitHubClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        assert await client.get_json("/items", use_cache=False) == {"number": 1}
        assert await client.get_json("/items", use_cache=False) == {"number": 2}
    assert not list(tmp_path.rglob("*.json"))
