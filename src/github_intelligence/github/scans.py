"""Enumeration evidence, not a promise of snapshot isolation or safe deletion."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ScanStatus(StrEnum):
    NOT_STARTED = "not_started"
    COMPLETE = "complete"
    LIMITED = "limited"
    TRUNCATED = "truncated"
    FAILED = "failed"


class PaginationCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: ScanStatus = ScanStatus.NOT_STARTED
    reason: str | None = None
    pages_fetched: int = 0
    items_seen: int = 0
    last_page: int | None = None
    next_page: int | None = 1
    last_source_id: str | None = None
    max_source_updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def observe(self, item: dict[str, Any]) -> None:
        self.items_seen += 1
        identifier = item.get("id", item.get("sha"))
        self.last_source_id = str(identifier) if identifier is not None else None
        stamp = item.get("updated_at") or item.get("submitted_at") or item.get("created_at")
        if stamp is None and isinstance(item.get("commit"), dict):
            stamp = (item["commit"].get("committer") or {}).get("date")
        if isinstance(stamp, str):
            try:
                parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(UTC)
                    if self.max_source_updated_at is None or parsed > self.max_source_updated_at:
                        self.max_source_updated_at = parsed
            except ValueError:
                pass  # Optional high-water hints must not invent timestamps.
