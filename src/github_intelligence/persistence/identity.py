"""Stable application identities independent of legacy retrieval chunk IDs."""

import hashlib
import json
import re
from datetime import UTC
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

from github_intelligence.models import DocumentType, GitHubDocument

HASH_VERSION = "sha256:normalized-v1"
_OPERATIONAL_KEYS = {
    "content_hash",
    "embedding_version",
    "chunker_version",
    "index_status",
    "last_seen_at",
    "ingested_at",
    "ingestion_job_id",
    "deleted_at",
}
_SECRET_KEYS = {
    "token",
    "github_token",
    "openai_api_key",
    "api_key",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "authorization",
    "database_url",
}


def reject_secrets(value: object) -> None:
    """Reject credential-bearing metadata keys instead of persisting their values."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in _SECRET_KEYS or normalized.endswith(
                ("_token", "_secret", "_password", "_api_key")
            ):
                raise ValueError("Credential-bearing metadata is not allowed")
            reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_secrets(item)


def canonical_repository(name: str) -> str:
    value = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*/[a-z0-9_.-]+", value):
        raise ValueError("Repository must use the GitHub owner/name format")
    if value.split("/")[1] in {".", ".."}:
        raise ValueError("Invalid repository name")
    return value


def source_key(document: GitHubDocument) -> str:
    if document.document_type != DocumentType.DOCUMENTATION:
        return document.github_id
    path = document.metadata.get("path")
    if not isinstance(path, str) or not path or any(p in {"", ".", ".."} for p in path.split("/")):
        raise ValueError("Documentation identity requires a relative metadata.path")
    branch = document.metadata.get("branch") or document.metadata.get("ref")
    if branch is None:
        url_path = unquote(urlsplit(str(document.source_url)).path)
        marker = "/blob/"
        suffix = "/" + path
        if marker not in url_path or not url_path.endswith(suffix):
            raise ValueError("Documentation identity requires a branch/ref or matching blob URL")
        branch = url_path.split(marker, 1)[1][: -len(suffix)]
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("Documentation identity requires a branch/ref")
    # JSON avoids ambiguous separators when branch names contain slashes.
    return json.dumps([branch, path], separators=(",", ":"), ensure_ascii=False)


def document_id(document: GitHubDocument) -> UUID:
    identity = [
        canonical_repository(document.repository),
        document.document_type.value,
        source_key(document),
    ]
    return uuid5(NAMESPACE_URL, json.dumps(identity, separators=(",", ":")))


def normalized_payload(document: GitHubDocument) -> dict[str, Any]:
    reject_secrets(document.metadata)
    payload = document.model_dump(mode="json")
    payload["repository"] = canonical_repository(document.repository)
    payload["metadata"] = {k: v for k, v in document.metadata.items() if k not in _OPERATIONAL_KEYS}
    for field in ("created_at", "updated_at"):
        timestamp = getattr(document, field)
        if timestamp is not None:
            if timestamp.tzinfo is None:
                raise ValueError("Source timestamps must include a timezone")
            payload[field] = timestamp.astimezone(UTC).isoformat()
    # Validate JSON serialization before opening a transaction.
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))  # type: ignore[no-any-return]


def content_hash(document: GitHubDocument) -> str:
    serialized = json.dumps(
        normalized_payload(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return HASH_VERSION + ":" + hashlib.sha256(serialized.encode()).hexdigest()
