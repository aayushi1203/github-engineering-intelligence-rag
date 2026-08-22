from pathlib import Path

from github_intelligence.github.cache import CacheEntry, DiskCache


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=60)
    cache.set("key", CacheEntry(body={"ok": True}, etag='"abc"'))
    result = cache.get("key")
    assert result is not None
    assert result.body == {"ok": True}
    assert result.etag == '"abc"'
    assert cache.is_fresh(result)

