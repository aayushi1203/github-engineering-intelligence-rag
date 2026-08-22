import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    body: Any
    etag: str | None = None
    stored_at: float = Field(default_factory=time.time)


class DiskCache:
    """Small dependency-free JSON cache suitable for GitHub API response caching."""

    def __init__(self, directory: Path, ttl_seconds: int) -> None:
        self.directory = directory
        self.ttl_seconds = ttl_seconds

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> CacheEntry | None:
        path = self._path(key)
        try:
            return CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return None

    def is_fresh(self, entry: CacheEntry) -> bool:
        return time.time() - entry.stored_at <= self.ttl_seconds

    def set(self, key: str, entry: CacheEntry) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entry.model_dump(mode="json"), handle, separators=(",", ":"))
            os.replace(temporary_name, self._path(key))
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

