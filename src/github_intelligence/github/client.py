import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from github_intelligence.config import Settings
from github_intelligence.github.cache import CacheEntry, DiskCache

logger = structlog.get_logger(__name__)


class GitHubError(RuntimeError):
    pass


class TransientGitHubError(GitHubError):
    pass


class RateLimitError(GitHubError):
    pass


class GitHubClient:
    """Async GitHub REST client with pagination, caching, retries, and rate-limit awareness."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.cache = DiskCache(settings.github_cache_dir, settings.github_cache_ttl_seconds)
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

    @staticmethod
    def _cache_key(url: str, params: Mapping[str, Any] | None) -> str:
        request = httpx.Request("GET", url, params=params)
        return str(request.url)

    async def get_json(
        self, url: str, params: Mapping[str, Any] | None = None, *, use_cache: bool = True
    ) -> Any:
        key = self._cache_key(url, params)
        cached = self.cache.get(key) if use_cache else None
        if cached and self.cache.is_fresh(cached):
            logger.debug("github_cache_hit", url=url)
            return cached.body

        headers = {"If-None-Match": cached.etag} if cached and cached.etag else {}
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.github_max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=8),
            retry=retry_if_exception_type((httpx.TransportError, TransientGitHubError)),
            reraise=True,
        ):
            with attempt:
                response = await self._client.get(url, params=params, headers=headers)
                await self._respect_rate_limit(response)
                if response.status_code == 304 and cached:
                    self.cache.set(key, CacheEntry(body=cached.body, etag=cached.etag))
                    return cached.body
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise TransientGitHubError(
                        f"GitHub returned transient status {response.status_code} for {url}"
                    )
                if response.status_code >= 400:
                    message = response.text[:500]
                    raise GitHubError(f"GitHub API {response.status_code} for {url}: {message}")
                body = response.json()
                if use_cache:
                    self.cache.set(key, CacheEntry(body=body, etag=response.headers.get("etag")))
                return body
        raise AssertionError("retry loop exited unexpectedly")

    async def _respect_rate_limit(self, response: httpx.Response) -> None:
        remaining = response.headers.get("x-ratelimit-remaining")
        reset = response.headers.get("x-ratelimit-reset")
        if response.status_code in {403, 429} and remaining == "0":
            wait_seconds = max(0, int(reset or "0") - int(time.time()))
            if wait_seconds <= 60:
                logger.warning("github_rate_limit_wait", wait_seconds=wait_seconds)
                await asyncio.sleep(wait_seconds + 1)
                raise TransientGitHubError("GitHub rate limit reached; retrying after reset")
            raise RateLimitError(
                f"GitHub rate limit exhausted; resets in approximately {wait_seconds} seconds"
            )

    async def paginate(
        self, url: str, params: Mapping[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        page = 1
        while not self.settings.github_max_pages or page <= self.settings.github_max_pages:
            page_params = dict(params or {}) | {"per_page": self.settings.github_per_page, "page": page}
            data = await self.get_json(url, page_params)
            if not isinstance(data, list):
                raise GitHubError(f"Expected a list from paginated endpoint {url}")
            for item in data:
                if isinstance(item, dict):
                    yield item
            if len(data) < self.settings.github_per_page:
                break
            page += 1

