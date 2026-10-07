import base64
import binascii
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import structlog

from github_intelligence.github import GitHubClient, GitHubError
from github_intelligence.github.scans import PaginationCheckpoint, ScanStatus
from github_intelligence.ingestion.models import IngestionReport, RepositoryMetadata, SourceScan
from github_intelligence.ingestion.normalizers import (
    normalize_comment,
    normalize_commit,
    normalize_documentation,
    normalize_issue,
    normalize_pull_request,
    normalize_release,
    normalize_review,
)
from github_intelligence.models import DocumentType, GitHubDocument
from github_intelligence.persistence.identity import canonical_repository, source_key

logger = structlog.get_logger(__name__)
_FAMILIES = [kind.value for kind in DocumentType]


class IngestionError(RuntimeError):
    def __init__(self, code: str, report: IngestionReport) -> None:
        self.code, self.report = code, report
        super().__init__(f"Repository ingestion failed: {code}; snapshot not published")


@dataclass(frozen=True, slots=True)
class IngestionOptions:
    include_comments: bool = True
    include_documentation: bool = True
    include_reviews: bool = True
    fresh: bool = True
    target_kind: str | None = None
    target_number: int | None = None
    expected_github_id: str | None = None
    allow_missing: bool = False


class RepositoryIngestor:
    def __init__(self, client: GitHubClient, doc_max_bytes: int = 1_000_000) -> None:
        self.client, self.doc_max_bytes = client, doc_max_bytes
        self.report = IngestionReport()
        self._active_family: str | None = None
        self._seen: set[tuple[str, str]] = set()
        self._options = IngestionOptions()

    async def _items(
        self, family: str, url: str, params: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        self._active_family = family
        checkpoint = PaginationCheckpoint()
        self.report.sources[family].checkpoints.append(checkpoint)
        try:
            async for item in self.client.paginate(
                url, params, fresh=self._options.fresh, checkpoint=checkpoint
            ):
                self._active_family = family
                yield item
        except Exception as error:
            self._active_family = family
            scan = self.report.sources[family]
            scan.status = ScanStatus.FAILED
            scan.reason = error.kind if isinstance(error, GitHubError) else "invalid_source"
            raise

    def _finish(self, family: str, *, parent: str | None = None) -> None:
        scan = self.report.sources[family]
        statuses = [cp.status for cp in scan.checkpoints]
        if parent:
            statuses.append(self.report.sources[parent].status)
        for status in (
            ScanStatus.FAILED,
            ScanStatus.NOT_STARTED,
            ScanStatus.TRUNCATED,
            ScanStatus.LIMITED,
        ):
            if status in statuses or (scan.status == status and scan.reason is not None):
                scan.status = status
                scan.reason = scan.reason or (
                    "parent_scope_incomplete" if parent else "endpoint_incomplete"
                )
                return
        scan.status = ScanStatus.COMPLETE

    def _document(self, document: GitHubDocument) -> GitHubDocument | None:
        repository = self.report.repository
        assert repository is not None
        document.metadata.update(
            {"repository_github_id": repository.github_id, "source_key": source_key(document)}
        )
        family = document.document_type.value
        key = (family, source_key(document))
        scan = self.report.sources[family]
        if key[1] not in scan.observed_source_keys:
            scan.observed_source_keys.append(key[1])
        if key in self._seen:
            # Offset pagination can repeat objects during concurrent source edits.
            scan.status, scan.reason = ScanStatus.LIMITED, "duplicate_during_scan"
            return None
        self._seen.add(key)
        scan.documents_emitted += 1
        return document

    async def ingest(
        self, repository: str, options: IngestionOptions | None = None
    ) -> AsyncIterator[GitHubDocument]:
        self._options = options or IngestionOptions()
        self._seen = set()
        self.report = IngestionReport(
            fresh=self._options.fresh, sources={family: SourceScan() for family in _FAMILIES}
        )
        self.report.limits = {
            "max_pages": self.client.settings.github_max_pages,
            "per_page": self.client.settings.github_per_page,
            "doc_max_bytes": self.doc_max_bytes,
        }
        self.report.sources["issue"].scope = {"state": "all"}
        self.report.sources["pull_request"].scope = {"state": "all"}
        for family in (
            "issue_comment",
            "pull_request_comment",
            "pull_request_review_comment",
            "pull_request_review",
        ):
            self.report.sources[family].scope = {
                "parents_from": "issue" if family == "issue_comment" else "pull_request"
            }
        excluded = []
        if not self._options.include_comments:
            excluded += ["issue_comment", "pull_request_comment", "pull_request_review_comment"]
        if not self._options.include_documentation:
            excluded += ["documentation"]
        if not self._options.include_reviews:
            excluded += ["pull_request_review"]
        self.report.requested_families = [family for family in _FAMILIES if family not in excluded]
        for family in excluded:
            self.report.sources[family].status = ScanStatus.LIMITED
            self.report.sources[family].reason = "excluded_by_options"
        target = self._options.target_kind
        target_families = {
            "issue": ["issue", "issue_comment"],
            "pull_request": [
                "pull_request",
                "pull_request_comment",
                "pull_request_review_comment",
                "pull_request_review",
            ],
            "release": ["release"],
            "default_push": ["commit", "documentation"],
        }
        if target is not None:
            if target not in target_families or (
                target != "default_push"
                and (not self._options.target_number or self._options.target_number < 1)
            ):
                raise ValueError("Invalid targeted ingestion scope")
            self.report.requested_families = [
                f for f in target_families[target] if f not in excluded
            ]
            for family, scan in self.report.sources.items():
                if family not in self.report.requested_families:
                    scan.status, scan.reason = ScanStatus.LIMITED, "outside_target_scope"
                else:
                    scan.scope = {
                        "target_kind": target,
                        "target_number": self._options.target_number,
                    }
        self._active_family = None
        try:
            requested = canonical_repository(repository)
            payload = await self.client.get_json(f"/repos/{requested}", fresh=self._options.fresh)
            if not isinstance(payload, dict) or not payload.get("id"):
                raise GitHubError("invalid_repository_metadata")
            repo = canonical_repository(payload["full_name"])
            if self._options.expected_github_id is not None and (
                str(payload["id"]) != self._options.expected_github_id or repo != requested
            ):
                raise GitHubError("repository_identity_conflict")
            self.report.repository = RepositoryMetadata(
                github_id=str(payload["id"]),
                requested_name=requested,
                name=repo,
                github_url=payload["html_url"],
                default_branch=payload["default_branch"],
                visibility=payload.get("visibility")
                or ("private" if payload.get("private") else "public"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
            )
            if target in {None, "default_push"}:
                # Delayed pushes refetch current head; do not follow a stale payload head.
                branch = quote(self.report.repository.default_branch, safe="")
                head = await self.client.get_json(
                    f"/repos/{repo}/commits/{branch}",
                    fresh=self._options.fresh,
                )
                head_sha, tree_sha = head["sha"], head["commit"]["tree"]["sha"]
                if not re.fullmatch(r"[0-9a-f]{40}", head_sha) or not re.fullmatch(
                    r"[0-9a-f]{40}", tree_sha
                ):
                    raise GitHubError("invalid_revision_metadata")
                self.report.repository.head_sha, self.report.repository.tree_sha = (
                    head_sha,
                    tree_sha,
                )
                self.report.sources["commit"].scope = {
                    "branch": self.report.repository.default_branch,
                    "head_sha": head_sha,
                }
            if target in {None, "issue"}:
                async for source_document in self._issues(repo):
                    yield source_document
            if target in {None, "pull_request"}:
                async for source_document in self._pull_requests(repo):
                    yield source_document
            if target in {None, "default_push"}:
                async for item in self._items(
                    "commit", f"/repos/{repo}/commits", {"sha": self.report.repository.head_sha}
                ):
                    doc = self._document(normalize_commit(repo, item))
                    if doc:
                        yield doc
                self._finish("commit")
            if target in {None, "release"}:
                async for item in self._parents("release", repo, "releases"):
                    doc = self._document(normalize_release(repo, item))
                    if doc:
                        yield doc
                self._finish("release")
            if self._options.include_documentation and target in {None, "default_push"}:
                async for source_document in self._documentation(repo):
                    yield source_document
            self.report.finished_at = datetime.now(UTC)
            logger.info("ingestion_finished", complete=self.report.complete_for_requested_scope)
        except Exception as error:
            code = (
                error.kind
                if isinstance(error, GitHubError)
                else "invalid_source_or_ingestion_failure"
            )
            if self._active_family:
                scan = self.report.sources[self._active_family]
                scan.status, scan.reason = ScanStatus.FAILED, code
            self.report.error_code, self.report.finished_at = code, datetime.now(UTC)
            if isinstance(error, GitHubError):
                self.report.failure_details = {
                    "status_code": error.status_code,
                    "attempts": error.attempts,
                    "retry_after": error.retry_after,
                    "exhausted": error.exhausted,
                    "deferred": error.deferred,
                }
            raise IngestionError(code, self.report) from None
        finally:
            # An abandoned async iterator can never claim a complete enumeration.
            if self.report.finished_at is None:
                self.report.finished_at = datetime.now(UTC)
                self.report.error_code = "scan_interrupted"
                if self._active_family:
                    scan = self.report.sources[self._active_family]
                    scan.status, scan.reason = ScanStatus.FAILED, "scan_interrupted"

    async def _parents(
        self, family: str, repo: str, endpoint: str
    ) -> AsyncIterator[dict[str, Any]]:
        if self._options.target_kind is None:
            async for item in self._items(
                family,
                f"/repos/{repo}/{endpoint}",
                {"state": "all"} if endpoint != "releases" else None,
            ):
                yield item
            return
        self._active_family = family
        endpoint_url = f"/repos/{repo}/{endpoint}/{self._options.target_number}"
        checkpoint = PaginationCheckpoint(endpoint=endpoint_url, started_at=datetime.now(UTC))
        self.report.sources[family].checkpoints.append(checkpoint)
        try:
            item = await self.client.get_json(
                f"/repos/{repo}/{endpoint}/{self._options.target_number}", fresh=True
            )
        except GitHubError as error:
            if (
                error.status_code == 404
                and self._options.allow_missing
                and family in {"issue", "release"}
            ):
                checkpoint.status = ScanStatus.COMPLETE
                checkpoint.reason = "explicit_deletion_target_missing"
                checkpoint.finished_at = datetime.now(UTC)
                checkpoint.next_page = None
                return  # Signed deletion evidence is separately authorized by the coordinator.
            raise
        if not isinstance(item, dict) or not item.get("id"):
            raise GitHubError("invalid_target_source")
        if family != "release" and item.get("number") != self._options.target_number:
            raise GitHubError("target_source_mismatch")
        if family == "release" and item["id"] != self._options.target_number:
            raise GitHubError("target_source_mismatch")
        if family == "issue" and "pull_request" in item:
            raise GitHubError("target_source_type_mismatch")
        checkpoint.observe(item)
        checkpoint.status = ScanStatus.COMPLETE
        checkpoint.finished_at = datetime.now(UTC)
        checkpoint.next_page = None
        yield item

    async def _issues(self, repo: str) -> AsyncIterator[GitHubDocument]:
        async for item in self._parents("issue", repo, "issues"):
            if "pull_request" in item:
                continue
            doc = self._document(normalize_issue(repo, item))
            if doc:
                yield doc
            if self._options.include_comments:
                # Enumerate even when the parent reports zero comments, to record that scope.
                async for comment in self._items(
                    "issue_comment", f"/repos/{repo}/issues/{item['number']}/comments"
                ):
                    doc = self._document(
                        normalize_comment(repo, comment, item["number"], pull_request=False)
                    )
                    if doc:
                        yield doc
        self._finish("issue")
        if self._options.include_comments:
            self._finish("issue_comment", parent="issue")

    async def _pull_requests(self, repo: str) -> AsyncIterator[GitHubDocument]:
        async for item in self._parents("pull_request", repo, "pulls"):
            doc = self._document(normalize_pull_request(repo, item))
            if doc:
                yield doc
            number = item["number"]
            if self._options.include_comments:
                for family, endpoint, review in [
                    ("pull_request_comment", f"/repos/{repo}/issues/{number}/comments", False),
                    ("pull_request_review_comment", f"/repos/{repo}/pulls/{number}/comments", True),
                ]:
                    async for comment in self._items(family, endpoint):
                        doc = self._document(
                            normalize_comment(
                                repo, comment, number, pull_request=True, review=review
                            )
                        )
                        if doc:
                            yield doc
            if self._options.include_reviews:
                async for review_body in self._items(
                    "pull_request_review", f"/repos/{repo}/pulls/{number}/reviews"
                ):
                    scan = self.report.sources["pull_request_review"]
                    key = str(review_body["id"])
                    if key not in scan.observed_source_keys:
                        scan.observed_source_keys.append(key)
                    if not str(review_body.get("body") or "").strip():
                        scan.skipped_empty += 1
                        continue
                    # Keep summaries separate; do not concatenate inline comments again.
                    doc = self._document(normalize_review(repo, review_body, number))
                    if doc:
                        yield doc
        self._finish("pull_request")
        for family in (
            "pull_request_comment",
            "pull_request_review_comment",
            "pull_request_review",
        ):
            if family in self.report.requested_families:
                self._finish(family, parent="pull_request")

    async def _documentation(self, repo: str) -> AsyncIterator[GitHubDocument]:
        self._active_family = "documentation"
        repository = self.report.repository
        assert repository is not None
        scan = self.report.sources["documentation"]
        scan.scope = {
            "branch": repository.default_branch,
            "head_sha": repository.head_sha,
            "tree_sha": repository.tree_sha,
        }
        tree = await self.client.get_json(
            f"/repos/{repo}/git/trees/{repository.tree_sha}",
            {"recursive": "1"},
            fresh=self._options.fresh,
        )
        if (
            not isinstance(tree, dict)
            or not isinstance(tree.get("tree"), list)
            or not isinstance(tree.get("truncated"), bool)
            or tree.get("sha") != repository.tree_sha
        ):
            raise GitHubError("invalid_tree_response")
        scan.status = ScanStatus.TRUNCATED if tree["truncated"] else ScanStatus.COMPLETE
        scan.reason = "recursive_tree_truncated" if tree["truncated"] else None
        for entry in tree["tree"]:
            if not isinstance(entry, dict):
                raise GitHubError("invalid_tree_entry")
            if not self._is_documentation(entry):
                continue
            path, sha = entry["path"], entry["sha"]
            if not re.fullmatch(r"[0-9a-f]{40}", sha):
                raise GitHubError("invalid_blob_revision")
            # Record membership even if content is empty or omitted by the size policy.
            identity_doc = normalize_documentation(
                repo,
                path,
                sha,
                f"https://github.com/{repo}/blob/{repository.head_sha}/{quote(path, safe='/')}",
                "(metadata only)",
                branch=repository.default_branch,
                commit_sha=repository.head_sha,
            )
            scan.observed_source_keys.append(source_key(identity_doc))
            if int(entry.get("size") or 0) > self.doc_max_bytes:
                scan.skipped_too_large += 1
                if scan.status != ScanStatus.TRUNCATED:
                    scan.status, scan.reason = ScanStatus.LIMITED, "doc_max_bytes"
                continue
            blob = await self.client.get_json(
                f"/repos/{repo}/git/blobs/{sha}", fresh=self._options.fresh
            )
            if not isinstance(blob, dict) or blob.get("sha") != sha:
                raise GitHubError("blob_revision_mismatch")
            content = self._decode_content(blob)
            if len(content.encode()) > self.doc_max_bytes:
                scan.skipped_too_large += 1
                if scan.status != ScanStatus.TRUNCATED:
                    scan.status, scan.reason = ScanStatus.LIMITED, "doc_max_bytes"
                continue
            if not content.strip():
                scan.skipped_empty += 1
                continue
            doc = self._document(identity_doc.model_copy(update={"text": content}))
            if doc:
                yield doc

    @staticmethod
    def _is_documentation(entry: dict[str, Any]) -> bool:
        if entry.get("type") != "blob":
            return False
        lower = str(entry.get("path", "")).lower()
        name = lower.rsplit("/", 1)[-1]
        return (
            name.startswith("readme")
            or lower.startswith(("docs/", "doc/", "documentation/"))
            or ("/docs/" in lower and lower.endswith((".md", ".mdx", ".rst", ".txt")))
        ) and lower.endswith((".md", ".mdx", ".rst", ".txt", "readme"))

    @staticmethod
    def _decode_content(payload: dict[str, Any]) -> str:
        encoded = payload.get("content")
        if payload.get("encoding") != "base64" or not isinstance(encoded, str):
            raise GitHubError("unsupported_blob_encoding")
        try:
            return base64.b64decode("".join(encoded.split()), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeError):
            raise GitHubError("invalid_documentation_encoding") from None
