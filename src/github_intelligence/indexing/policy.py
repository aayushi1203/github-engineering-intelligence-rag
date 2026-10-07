"""Deletion is opt-in, scope-specific and restricted to immutable documentation trees."""

import hashlib
import json
import re
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from github_intelligence.github.scans import ScanStatus
from github_intelligence.ingestion.models import IngestionReport
from github_intelligence.ingestion.service import RepositoryIngestor

IDENTITY_VERSION = "application-chunk-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def chunk_identity(
    repository_id: UUID,
    family: str,
    key: str,
    ordinal: int,
    payload_hash: str,
    embedding_version: str,
    chunker_version: str,
) -> str:
    # Content versions are immutable so a pending write cannot corrupt a READY revision.
    return str(
        uuid5(
            NAMESPACE_URL,
            json.dumps(
                [
                    IDENTITY_VERSION,
                    str(repository_id),
                    family,
                    key,
                    ordinal,
                    payload_hash,
                    embedding_version,
                    chunker_version,
                ],
                separators=(",", ":"),
            ),
        )
    )


def deletion_reason(
    report: IngestionReport, family: str, old_scope: dict[str, Any], key: str, *, reconcile: bool
) -> str | None:
    if not reconcile:
        return "reconciliation_disabled"
    scan = report.sources.get(family)
    if family not in report.requested_families or scan is None:
        return "family_excluded"
    if scan.status != ScanStatus.COMPLETE:
        return "scan_" + scan.status.value.lower()
    if key in scan.observed_source_keys:
        return "source_observed_without_document"
    if family != "documentation":
        return "mutable_pagination_not_authoritative"
    if not report.fresh:
        return "cached_scan_not_authoritative"
    repository = report.repository
    scope = scan.scope
    if (
        repository is None
        or not re.fullmatch(r"[0-9a-f]{40}", repository.head_sha or "")
        or not re.fullmatch(r"[0-9a-f]{40}", repository.tree_sha or "")
        or not repository.head_sha
        or not repository.tree_sha
        or scope.get("head_sha") != repository.head_sha
        or scope.get("tree_sha") != repository.tree_sha
        or scope.get("branch") != repository.default_branch
    ):
        return "immutable_revision_missing"
    try:
        branch, path = json.loads(key)
    except (ValueError, TypeError):
        return "invalid_documentation_identity"
    if (
        branch != scope.get("branch")
        or old_scope.get("branch") != branch
        or not isinstance(path, str)
        or not path
    ):
        return "scope_mismatch"
    known_scope = {"branch", "head_sha", "tree_sha"}
    if set(scope) - known_scope or set(old_scope) - known_scope:
        return "unknown_scope_filter"
    if not RepositoryIngestor._is_documentation({"type": "blob", "path": path}):
        return "path_outside_documentation_scope"
    if old_scope.get("tree_sha") == scope.get("tree_sha"):
        return "unchanged_tree_membership_conflict"
    if any(checkpoint.status != ScanStatus.COMPLETE for checkpoint in scan.checkpoints):
        return "ambiguous_checkpoint"
    if len(set(scan.observed_source_keys)) != len(scan.observed_source_keys):
        return "ambiguous_membership"
    if scan.skipped_too_large or scan.reason is not None:
        return "ambiguous_scan"
    return None
