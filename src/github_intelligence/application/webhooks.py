"""Authenticated receipt, narrow canonical refresh scopes, and durable deduplication."""

import hashlib
import hmac
import json
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from github_intelligence.application.jobs import enqueue
from github_intelligence.ingestion.models import IngestionReport
from github_intelligence.persistence.identity import canonical_repository
from github_intelligence.persistence.models import (
    Document,
    IngestionJob,
    Repository,
    WebhookDelivery,
)

SUPPORTED = {
    "issues",
    "issue_comment",
    "pull_request",
    "pull_request_review",
    "pull_request_review_comment",
    "push",
    "release",
}


def verify_signature(raw: bytes, signature: str | None, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.encode(), (signature or "").encode())


def positive_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Invalid native identity")
    return str(value)


def refresh_target(
    event: str, body: dict[str, Any], repo: Repository
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if event == "issues" and ("comment" in body or "review" in body or "pull_request" in body):
        raise ValueError("Ambiguous issue event shape")
    if event == "push":
        ref = body.get("ref")
        after = body.get("after")
        if (
            not isinstance(ref, str)
            or len(ref) > 1024
            or not re.fullmatch(r"[0-9a-f]{40}", after or "")
        ):
            raise ValueError("Invalid push identity")
        metadata = {
            "ref": ref,
            "after": after,
            "forced": body.get("forced") is True,
            "deleted": body.get("deleted") is True,
        }
        # Registered/indexed branch, never the potentially stale payload default_branch.
        if not repo.default_branch or ref != "refs/heads/" + repo.default_branch:
            return {"kind": "ignored_push"}, metadata
        return {"kind": "default_push"}, metadata
    if event == "release":
        item = body["release"]
        number = int(positive_id(item["id"]))
        target = {"kind": "release", "number": number}
        family = "release"
    else:
        item = body["issue"] if event in {"issues", "issue_comment"} else body["pull_request"]
        number = int(positive_id(item["number"]))
        is_pr = event not in {"issues", "issue_comment"} or "pull_request" in item
        target = {"kind": "pull_request" if is_pr else "issue", "number": number}
        family = "pull_request" if is_pr else "issue"
        if event in {"issue_comment", "pull_request_review_comment"}:
            family = (
                "pull_request_review_comment"
                if event == "pull_request_review_comment"
                else "pull_request_comment"
                if is_pr
                else "issue_comment"
            )
            item = body["comment"]
        elif event == "pull_request_review":
            item = body["review"]
            family = "pull_request_review"
    metadata["source_id"] = positive_id(item["id"])
    timestamp = item.get("updated_at") or item.get("submitted_at") or item.get("created_at")
    if isinstance(timestamp, str) and len(timestamp) <= 64:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                metadata["source_updated_at"] = parsed.astimezone(UTC).isoformat()
        except ValueError:
            pass  # Diagnostic only; never an ordering authority.
    metadata["target"] = target
    # Closure/dismissal are updates. Only these documented deletion events qualify.
    if body.get("action") == "deleted" and event in {
        "issues",
        "issue_comment",
        "pull_request_review_comment",
        "release",
    }:
        metadata["deletion"] = {
            "family": family,
            "key": metadata["source_id"],
            "parent_number": number,
        }
    return target, metadata


def receive(
    session: Session,
    *,
    delivery_id: str,
    event: str,
    raw: bytes,
    configuration: dict[str, Any],
) -> tuple[WebhookDelivery, bool]:
    digest = hashlib.sha256(raw).hexdigest()
    inserted = session.scalar(
        insert(WebhookDelivery)
        .values(
            delivery_id=delivery_id,
            event_type=event,
            body_sha256=digest,
            status="FAILED",
            error_code="malformed_payload",
        )
        .on_conflict_do_nothing(index_elements=["delivery_id"])
        .returning(WebhookDelivery.delivery_id)
    )
    delivery = session.get(WebhookDelivery, delivery_id)
    assert delivery is not None
    if inserted is None:
        if delivery.body_sha256 != digest or delivery.event_type != event:
            raise ValueError("Delivery identifier reused with different content")
        return delivery, True
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("Expected object")
        if event not in SUPPORTED:
            delivery.status, delivery.error_code = "IGNORED", "unsupported_event"
            # Repository rename events themselves are unsupported for indexing, but
            # native identity conflicts still need an operator-visible deferral.
            repository = body.get("repository")
            if isinstance(repository, dict):
                try:
                    native = positive_id(repository.get("id"))
                    name = canonical_repository(repository.get("full_name", ""))
                except (ValueError, TypeError, AttributeError):
                    return delivery, False
                delivery.repository_github_id = native
                known = session.scalars(
                    select(Repository).where(Repository.github_id == native).with_for_update()
                ).one_or_none()
                if known is not None:
                    delivery.repository_id = known.id
                    if known.name != name:
                        delivery.status, delivery.error_code = (
                            "DEFERRED",
                            "repository_rename_requires_operator",
                        )
            return delivery, False
        action = body.get("action")
        if event != "push" and (
            not isinstance(action, str) or not re.fullmatch(r"[a-z_]{1,64}", action)
        ):
            raise ValueError("Invalid action")
        delivery.action = action if event != "push" else None
        native = positive_id(body["repository"]["id"])
        raw_name = body["repository"]["full_name"]
        if not isinstance(raw_name, str):
            raise ValueError("Invalid repository name type")
        name = canonical_repository(raw_name)
        delivery.repository_github_id = native
        # Resolve by native ID first; a rename must not create another repository.
        repo = session.scalars(
            select(Repository).where(Repository.github_id == native).with_for_update()
        ).one_or_none()
        if repo is not None and repo.name != name:
            delivery.repository_id = repo.id
            delivery.status, delivery.error_code = "DEFERRED", "repository_rename_requires_operator"
            return delivery, False
        if repo is None:
            repo = session.scalars(
                select(Repository).where(Repository.name == name).with_for_update()
            ).one_or_none()
        if repo is None:
            delivery.status, delivery.error_code = "IGNORED", "repository_not_registered"
            return delivery, False
        delivery.repository_id = repo.id
        if repo.github_id not in {None, native}:
            delivery.status, delivery.error_code = "DEFERRED", "repository_identity_conflict"
            return delivery, False
        target, metadata = refresh_target(event, body, repo)
        delivery.event_metadata = metadata
        if target["kind"] == "ignored_push":
            delivery.status, delivery.error_code = "IGNORED", "branch_not_indexed"
            return delivery, False
        target["github_id"] = native
        # Receipt's repository lock serializes enqueue/coalescing. Never mutate a claimed job.
        pending = session.scalars(
            select(IngestionJob)
            .where(
                IngestionJob.repository_id == repo.id,
                IngestionJob.status == "QUEUED",
                IngestionJob.attempts == 0,
                IngestionJob.scope["configuration"] == configuration,
                IngestionJob.scope["webhook_target"] == target,
            )
            .with_for_update(skip_locked=True)
            .order_by(IngestionJob.created_at)
        ).first()
        if pending is None:
            pending = enqueue(
                session,
                repo.id,
                key="webhook:" + str(uuid4()),
                scope={
                    "configuration": configuration,
                    "reconcile": target["kind"] == "default_push",
                    "webhook_target": target,
                    "options": {
                        "fresh": True,
                        "target_kind": target["kind"],
                        "target_number": target.get("number"),
                        "expected_github_id": native,
                        "allow_missing": "deletion" in metadata,
                    },
                },
            )
        elif "deletion" in metadata and not pending.scope["options"].get("allow_missing"):
            scope = dict(pending.scope)
            scope["options"] = {**scope["options"], "allow_missing": True}
            pending.scope = scope
        delivery.job_id, delivery.status, delivery.error_code = pending.id, "ACCEPTED", None
    except (ValueError, KeyError, TypeError, UnicodeError, RecursionError):
        delivery.status, delivery.error_code = "FAILED", "malformed_payload"
    return delivery, False


def authorized_deletions(session: Session, job_id: UUID, report: IngestionReport) -> set[str]:
    """Authorize only persisted signed deletion evidence verified by a fresh canonical scope."""
    from github_intelligence.github.scans import ScanStatus
    from github_intelligence.indexing.planner import identity_key

    job = session.get(IngestionJob, job_id)
    if job is None or "webhook_target" not in job.scope or report.repository is None:
        raise ValueError("Deletion authorization requires a webhook job")
    repo = session.get(Repository, job.repository_id)
    if (
        repo is None
        or repo.name != report.repository.name
        or report.repository.github_id != job.scope["webhook_target"]["github_id"]
    ):
        raise ValueError("Canonical repository identity did not match delivery")
    allowed: set[str] = set()
    for delivery in session.scalars(
        select(WebhookDelivery).where(WebhookDelivery.job_id == job_id)
    ):
        deletion = delivery.event_metadata.get("deletion")
        if not deletion or delivery.repository_github_id != report.repository.github_id:
            continue
        family, key = deletion["family"], deletion["key"]
        scan = report.sources.get(family)
        if (
            not report.fresh
            or scan is None
            or family not in report.requested_families
            or scan.status != ScanStatus.COMPLETE
            or scan.reason is not None
            or key in scan.observed_source_keys
            or scan.scope.get("target_number") != deletion["parent_number"]
        ):
            continue
        document = session.scalars(
            select(Document).where(
                Document.repository_id == repo.id,
                Document.source_type == family,
                Document.source_key == key,
            )
        ).one_or_none()
        if document is None:
            continue
        parent = document.normalized_payload["metadata"].get(
            "parent_number", document.normalized_payload["metadata"].get("number")
        )
        if family == "release" or parent == deletion["parent_number"]:
            allowed.add(identity_key(family, key))
    return allowed
