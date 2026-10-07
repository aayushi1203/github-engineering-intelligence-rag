from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from github_intelligence.github.scans import PaginationCheckpoint, ScanStatus


class RepositoryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    github_id: str
    requested_name: str
    name: str
    github_url: HttpUrl
    default_branch: str
    visibility: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    head_sha: str | None = None
    tree_sha: str | None = None


class SourceScan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: ScanStatus = ScanStatus.NOT_STARTED
    reason: str | None = None
    documents_emitted: int = 0
    skipped_empty: int = 0
    skipped_too_large: int = 0
    checkpoints: list[PaginationCheckpoint] = Field(default_factory=list)
    # Stable keys, including deliberately omitted empty/oversized sources.
    observed_source_keys: list[str] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)


class IngestionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    scan_id: UUID = Field(default_factory=uuid4)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    repository: RepositoryMetadata | None = None
    requested_families: list[str] = Field(default_factory=list)
    sources: dict[str, SourceScan] = Field(default_factory=dict)
    fresh: bool = True
    consistency: Literal["best_effort"] = "best_effort"
    limits: dict[str, int] = Field(default_factory=dict)
    snapshot_sha256: str | None = None
    error_code: str | None = None
    failure_details: dict[str, Any] = Field(default_factory=dict)

    @property
    def complete_for_requested_scope(self) -> bool:
        return (
            self.error_code is None
            and self.finished_at is not None
            and all(
                self.sources[family].status == ScanStatus.COMPLETE
                for family in self.requested_families
            )
        )

    @property
    def full_snapshot(self) -> bool:
        return self.complete_for_requested_scope and all(
            scan.status == ScanStatus.COMPLETE for scan in self.sources.values()
        )
