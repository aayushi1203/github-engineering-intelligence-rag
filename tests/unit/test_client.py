from pathlib import Path

import httpx
import pytest

from github_intelligence.config import Settings
from github_intelligence.github.client import GitHubClient


@pytest.mark.asyncio
async def test_paginate_and_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        page = request.url.params["page"]
        body = [{"id": 1}, {"id": 2}] if page == "1" else []
        return httpx.Response(200, json=body, headers={"etag": f'"{page}"'}, request=request)

    settings = Settings(
        github_cache_dir=tmp_path, github_cache_ttl_seconds=3600, github_per_page=2
    )
    async with GitHubClient(settings, transport=httpx.MockTransport(handler)) as client:
        first = [item async for item in client.paginate("/items")]
        second = [item async for item in client.paginate("/items")]
    assert first == second == [{"id": 1}, {"id": 2}]
    assert calls == 2

