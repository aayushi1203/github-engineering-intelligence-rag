import asyncio
import hashlib
import math
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import structlog

from github_intelligence.config import Settings
from github_intelligence.github.cache import CacheEntry, DiskCache
from github_intelligence.github.scans import PaginationCheckpoint, ScanStatus

logger = structlog.get_logger(__name__)


class GitHubError(RuntimeError):
    """Sanitized boundary failure; response bodies and credentials are never included."""

    def __init__(
        self,
        kind: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        attempts: int = 1,
    ) -> None:
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.attempts = attempts
        self.deferred = False
        self.exhausted = False
        super().__init__(kind)

    def __str__(self) -> str:
        status = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        delay = (
            f"; retry no earlier than {self.retry_after:.1f}s"
            if self.retry_after is not None
            else ""
        )
        exhausted = "; retry attempts exhausted" if self.exhausted else ""
        return f"GitHub {self.kind}{status} after {self.attempts} attempt(s){delay}{exhausted}"


class TransientGitHubError(GitHubError):
    pass


class RateLimitError(TransientGitHubError):
    pass


@dataclass
class _ResponseData:
    body: Any
    link_present: bool = False
    next_page: int | None = None


class GitHubClient:
    """Bounded async REST requests with isolated caches and explicit fresh reads."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self._sleep, self._clock = sleep, clock
        # A versioned credential/API namespace ignores unsafe legacy cache entries.
        identity = (
            settings.github_api_url.rstrip("/") + "\0" + (settings.github_token or "anonymous")
        )
        namespace = hashlib.sha256(identity.encode()).hexdigest()
        self.cache = DiskCache(
            settings.github_cache_dir / "v2" / namespace, settings.github_cache_ttl_seconds
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-engineering-intelligence/0.1",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        self._client = httpx.AsyncClient(
            base_url=settings.github_api_url,
            headers=headers,
            timeout=settings.github_request_timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _request(self, url: str, params: Mapping[str, Any] | None) -> httpx.Request:
        request = self._client.build_request("GET", url, params=params)
        base = self._client.base_url
        if (request.url.scheme, request.url.host, request.url.port) != (
            base.scheme,
            base.host,
            base.port,
        ):
            raise GitHubError("cross_origin_request_rejected")
        if (
            request.url.username
            or request.url.password
            or any(
                key.lower() in {"access_token", "token", "authorization", "api_key"}
                for key in request.url.params
            )
        ):
            raise GitHubError("credential_url_rejected")
        return request

    def _retry_after(self, value: str | None) -> float | None:
        if value is None:
            return None
        try:
            delay = float(value)
            if math.isfinite(delay) and delay >= 0:
                return delay
        except ValueError:
            pass
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            return max(0.0, date.timestamp() - self._clock())
        except (TypeError, ValueError, OverflowError):
            return None

    def _failure(self, response: httpx.Response) -> GitHubError | None:
        status = response.status_code
        retry_after = self._retry_after(response.headers.get("retry-after"))
        if status in {403, 429}:
            if response.headers.get("x-ratelimit-remaining") == "0":
                reset = None
                try:
                    reset = max(0.0, float(response.headers["x-ratelimit-reset"]) - self._clock())
                    if not math.isfinite(reset):
                        reset = None
                except (KeyError, ValueError):
                    pass
                return RateLimitError(
                    "primary_rate_limit",
                    status_code=status,
                    retry_after=max(
                        retry_after if retry_after is not None else 0,
                        reset
                        if reset is not None
                        else (retry_after if retry_after is not None else 60),
                    ),
                )
            # Error messages are inspected in memory solely to classify rate limiting.
            try:
                data = response.json()
                message = str(data.get("message", "")).lower() if isinstance(data, dict) else ""
            except ValueError:
                message = ""
            if (
                status == 429
                or retry_after is not None
                or any(phrase in message for phrase in ("secondary rate", "abuse", "rate limit"))
            ):
                return RateLimitError(
                    "secondary_rate_limit",
                    status_code=status,
                    retry_after=retry_after if retry_after is not None else 60,
                )
        if status in {500, 502, 503, 504}:
            return TransientGitHubError(
                "transient_http_failure", status_code=status, retry_after=retry_after
            )
        if status >= 400:
            return GitHubError("request_rejected", status_code=status)
        return None

    def _pagination_links(self, response: httpx.Response) -> tuple[bool, int | None]:
        next_url = response.links.get("next", {}).get("url")
        if next_url is None:
            return "link" in response.headers, None
        request = self._request(next_url, None)
        if request.url.path != response.request.url.path:
            raise GitHubError("invalid_pagination_link")
        try:
            page = int(request.url.params["page"])
            previous = int(response.request.url.params.get("page", "1"))
            if page != previous + 1:
                raise ValueError
        except (ValueError, KeyError):
            raise GitHubError("invalid_pagination_link") from None
        return True, page

    async def _send(
        self, url: str, params: Mapping[str, Any] | None, headers: dict[str, str]
    ) -> httpx.Response:
        target = self._request(url, params).url
        for _ in range(4):
            response = await self._client.get(target, headers=headers)
            if response.status_code not in {301, 302, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                raise GitHubError("redirect_without_location")
            target = self._request(str(response.request.url.join(location)), None).url
        raise GitHubError("redirect_limit_exceeded")

    async def _get_data(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        use_cache: bool = True,
        fresh: bool = False,
    ) -> _ResponseData:
        request = self._request(url, params)
        key = str(request.url)
        cached = self.cache.get(key) if use_cache else None
        if cached and not fresh and self.cache.is_fresh(cached):
            logger.debug("github_cache_hit")
            return _ResponseData(cached.body, cached.link_present, cached.next_page)
        headers = {"If-None-Match": cached.etag} if cached and cached.etag else {}
        for attempt in range(1, self.settings.github_max_retries + 1):
            try:
                response = await self._send(url, params, headers)
                error = self._failure(response)
                if error is not None:
                    raise error
                if response.status_code == 304:
                    if cached is None:
                        raise GitHubError("unexpected_not_modified")
                    link_present, next_page = self._pagination_links(response)
                    # An unchanged full page may gain a next page. Fresh reads cannot
                    # trust old navigation headers when a 304 omits them.
                    if not fresh and not link_present:
                        link_present, next_page = cached.link_present, cached.next_page
                    self.cache.set(
                        key,
                        cached.model_copy(
                            update={
                                "stored_at": self._clock(),
                                "link_present": link_present,
                                "next_page": next_page,
                            }
                        ),
                    )
                    return _ResponseData(cached.body, link_present, next_page)
                try:
                    body = response.json()
                except ValueError:
                    raise GitHubError("invalid_json_response") from None
                link_present, next_page = self._pagination_links(response)
                if use_cache:
                    self.cache.set(
                        key,
                        CacheEntry(
                            body=body,
                            etag=response.headers.get("etag"),
                            link_present=link_present,
                            next_page=next_page,
                        ),
                    )
                return _ResponseData(body, link_present, next_page)
            except httpx.TransportError:
                error = TransientGitHubError("transport_failure", attempts=attempt)
            except GitHubError as failure:
                error = failure
            error.attempts = attempt
            if (
                not isinstance(error, TransientGitHubError)
                or attempt == self.settings.github_max_retries
            ):
                error.exhausted = isinstance(error, TransientGitHubError)
                raise error from None
            exponential = min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.5)
            if isinstance(error, RateLimitError):
                minimum = error.retry_after if error.retry_after is not None else 60
                delay = max(exponential, minimum * 2 ** (attempt - 1))
            else:
                delay = max(exponential, error.retry_after or 0)
            if delay > self.settings.github_max_retry_wait_seconds:
                error.deferred = True
                raise error from None  # Never shorten GitHub's minimum wait to fit our budget.
            logger.warning("github_retry", attempt=attempt, kind=error.kind, wait_seconds=delay)
            await self._sleep(delay)
        raise AssertionError("bounded retry loop exited unexpectedly")

    async def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        use_cache: bool = True,
        fresh: bool = False,
    ) -> Any:
        return (await self._get_data(url, params, use_cache=use_cache, fresh=fresh)).body

    async def paginate(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        fresh: bool = False,
        checkpoint: PaginationCheckpoint | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        from datetime import datetime

        scan = checkpoint if checkpoint is not None else PaginationCheckpoint()
        scan.endpoint = self._request(url, params).url.path
        scan.parameters = dict(params or {}) | {"per_page": self.settings.github_per_page}
        scan.started_at = datetime.now(UTC)
        scan.status, scan.reason = ScanStatus.FAILED, "not_completed"
        page = 1
        try:
            while True:
                data = await self._get_data(
                    url, dict(scan.parameters) | {"page": page}, fresh=fresh
                )
                if not isinstance(data.body, list) or any(
                    not isinstance(item, dict) for item in data.body
                ):
                    raise GitHubError("invalid_paginated_response")
                scan.pages_fetched += 1
                scan.last_page = page
                has_next = data.next_page is not None or (
                    not data.link_present and len(data.body) >= self.settings.github_per_page
                )
                scan.next_page = data.next_page or (page + 1 if has_next else None)
                for item in data.body:
                    scan.observe(item)
                    yield item
                if not has_next:
                    scan.status, scan.reason = ScanStatus.COMPLETE, None
                    break
                if self.settings.github_max_pages and page >= self.settings.github_max_pages:
                    scan.status, scan.reason = ScanStatus.LIMITED, "max_pages"
                    break
                page = scan.next_page or page + 1
        except GitHubError as error:
            scan.reason = error.kind
            raise
        finally:
            scan.finished_at = datetime.now(UTC)
