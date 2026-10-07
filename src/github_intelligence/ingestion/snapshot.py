"""Publish JSONL only after successful enumeration; bind its report by checksum."""

import hashlib
import os
import tempfile
from pathlib import Path

from github_intelligence.ingestion.models import IngestionReport
from github_intelligence.ingestion.service import (
    IngestionError,
    IngestionOptions,
    RepositoryIngestor,
)


def report_path(output: Path) -> Path:
    return output.with_name(output.name + ".scan.json")


def failed_report_path(output: Path) -> Path:
    return output.with_name(output.name + ".failed-scan.json")


def load_snapshot_report(output: Path) -> IngestionReport:
    report = IngestionReport.model_validate_json(report_path(output).read_text())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if (
        report.snapshot_sha256 != digest
        or report.error_code is not None
        or report.finished_at is None
    ):
        raise ValueError("Snapshot/report publication is incomplete or mismatched")
    return report


def _publish_report(path: Path, report: IngestionReport) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".scan-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


async def publish_snapshot(
    ingestor: RepositoryIngestor,
    repository: str,
    output: Path,
    options: IngestionOptions | None = None,
    *,
    allow_incomplete: bool = False,
) -> IngestionReport:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=".ingestion-", suffix=".tmp")
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as handle:
            async for document in ingestor.ingest(repository, options):
                line = (document.model_dump_json() + "\n").encode()
                digest.update(line)
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        report = ingestor.report
        if not report.complete_for_requested_scope and not allow_incomplete:
            raise IngestionError("incomplete_requested_scope", report)
        if report.error_code is not None:
            raise IngestionError(report.error_code, report)
        report.snapshot_sha256 = digest.hexdigest()
        # Two paths cannot be renamed atomically. Publish report first and data last;
        # a crash between them is detected by load_snapshot_report, never deletion evidence.
        _publish_report(report_path(output), report)
        os.replace(temporary, output)
        return report
    except (IngestionError, OSError, ValueError) as error:
        report = error.report if isinstance(error, IngestionError) else ingestor.report
        code = error.code if isinstance(error, IngestionError) else "publication_failure"
        diagnostic = report.model_copy(update={"error_code": code, "snapshot_sha256": None})
        try:
            _publish_report(failed_report_path(output), diagnostic)
        except OSError:
            pass  # A disk failure must not disguise the original failed publication.
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
