"""Serialized, replayable copy-on-write indexing with PostgreSQL publication."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from github_intelligence.indexing.models import ChunkManifest, IndexHead, IndexOperation
from github_intelligence.indexing.planner import identity_key, metadata_state, prepare_plan
from github_intelligence.indexing.policy import digest
from github_intelligence.indexing.storage import IncrementalVectorStore, file_hash, publish_lexical
from github_intelligence.ingestion.models import IngestionReport
from github_intelligence.ingestion.snapshot import load_snapshot_report
from github_intelligence.models import DocumentChunk, DocumentType, GitHubDocument
from github_intelligence.persistence.identity import (
    canonical_repository,
    content_hash,
    reject_secrets,
    source_key,
)
from github_intelligence.persistence.models import Document, IngestionJob, Repository
from github_intelligence.persistence.operations import register_repository, upsert_document
from github_intelligence.retrieval import BM25Retriever, DocumentChunker
from github_intelligence.retrieval.vector import Encoder, VectorMatch


class IndexingError(RuntimeError):
    """Sanitized failure with retry classification, without exposing its cause."""

    def __init__(
        self, message: str, *, code: str = "indexing_invariant", retryable: bool = False
    ) -> None:
        self.code, self.retryable = code, retryable
        super().__init__(message)


class IncrementalCoordinator:
    def __init__(
        self,
        engine: Engine,
        store: IncrementalVectorStore,
        encoder: Encoder,
        root: Path,
        *,
        embedding_version: str,
        chunker: DocumentChunker | None = None,
        boundary: Callable[[str], None] | None = None,
        publication_guard: Callable[[Session, UUID], None] | None = None,
    ) -> None:
        if not embedding_version.strip():
            raise ValueError("An explicit embedding version is required")
        self.engine, self.store, self.encoder = engine, store, encoder
        self.root = root.resolve()
        self.chunker = chunker or DocumentChunker()
        self.embedding_version = embedding_version
        self.chunker_version = f"structure-v1:{self.chunker.max_words}:{self.chunker.overlap_words}"
        self.boundary = boundary or (lambda stage: None)
        self.publication_guard = publication_guard
        self.configuration = {
            "root": str(self.root),
            "qdrant_path": store.identity,
            "collection": store.collection,
            "embedding_version": embedding_version,
            "chunker_version": self.chunker_version,
        }

    def run(
        self,
        snapshot: Path,
        *,
        repository: str,
        reconcile: bool = False,
        webhook_job_id: UUID | None = None,
    ) -> IndexOperation:
        report = load_snapshot_report(snapshot)
        raw = snapshot.read_bytes()
        if hashlib.sha256(raw).hexdigest() != report.snapshot_sha256:
            raise ValueError("Snapshot changed during validation")
        documents = [
            GitHubDocument.model_validate_json(line) for line in raw.splitlines() if line.strip()
        ]
        name = canonical_repository(repository)
        self._validate(report, documents, name)
        reject_secrets(report.model_dump(mode="json"))
        config = {**self.configuration, "reconcile": reconcile}
        if webhook_job_id is not None:
            config["webhook_job_id"] = str(webhook_job_id)
        request = digest([str(report.scan_id), report.snapshot_sha256, config])
        # Session-level lock survives each progress commit. Disconnect/crash releases it.
        lock = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)
        with self.engine.connect() as connection:
            locked: bool = connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": lock}
            ).scalar_one()
            connection.commit()
            if not locked:
                raise IndexingError(
                    "Another indexing operation holds this repository lock",
                    code="repository_busy",
                    retryable=True,
                )
            try:
                with Session(connection, expire_on_commit=False) as session:
                    operation = self._operation(session, name, report, request, config)
                    if operation.status == "READY":
                        return operation
                    try:
                        explicit_deletions: set[str] = set()
                        if webhook_job_id is not None:
                            from github_intelligence.application.webhooks import (
                                authorized_deletions,
                            )

                            explicit_deletions = authorized_deletions(
                                session, webhook_job_id, report
                            )
                        prepare_plan(
                            session,
                            operation,
                            documents,
                            report,
                            reconcile,
                            chunker=self.chunker,
                            embedding_version=self.embedding_version,
                            chunker_version=self.chunker_version,
                            boundary=self.boundary,
                            explicit_deletions=explicit_deletions,
                        )
                        self._execute(session, operation)
                        return operation
                    except Exception as error:
                        from github_intelligence.application.failures import classify

                        failure = classify(error)
                        stage = operation.status if operation.status != "READY" else "VERIFYING"
                        session.rollback()
                        failed_operation = session.get(IndexOperation, operation.id)
                        assert failed_operation is not None
                        operation = failed_operation
                        operation.status = "FAILED"
                        operation.error_code = failure.code
                        operation.failure_stage = stage
                        session.commit()
                        raise IndexingError(
                            f"Indexing revision {operation.id} failed. "
                            "Previous READY is preserved. "
                            "Inspect failure classification before retrying this snapshot.",
                            code=failure.code,
                            retryable=failure.retryable,
                        ) from None
            finally:
                connection.rollback()
                connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock})
                connection.commit()

    def _validate(
        self, report: IngestionReport, documents: list[GitHubDocument], name: str
    ) -> None:
        repo = report.repository
        if report.schema_version != 1 or not set(report.sources) <= {t.value for t in DocumentType}:
            raise ValueError("Unsupported scan schema/source family")
        if repo is None or canonical_repository(repo.name) != name or not repo.github_id:
            raise ValueError("Validated report must identify the explicit repository")
        if not set(report.requested_families) <= set(report.sources):
            raise ValueError("Report source scope is incomplete")
        seen: set[str] = set()
        counts: dict[str, int] = {}
        for doc in documents:
            family, key = doc.document_type.value, source_key(doc)
            identity = identity_key(family, key)
            scan = report.sources.get(family)
            if (
                canonical_repository(doc.repository) != name
                or identity in seen
                or family not in report.requested_families
                or scan is None
                or key not in scan.observed_source_keys
                or doc.metadata.get("repository_github_id") != repo.github_id
                or doc.metadata.get("source_key") != key
            ):
                raise ValueError("Snapshot document identity/membership conflicts with report")
            if family == "documentation" and (
                doc.metadata.get("branch") != scan.scope.get("branch")
                or doc.metadata.get("commit_sha") != scan.scope.get("head_sha")
            ):
                raise ValueError("Documentation document conflicts with revision scope")
            content_hash(doc)
            seen.add(identity)
            counts[family] = counts.get(family, 0) + 1
        for family, scan in report.sources.items():
            if scan.documents_emitted != counts.get(family, 0):
                raise ValueError("Report document counts do not match snapshot")

    def _operation(
        self,
        session: Session,
        name: str,
        report: IngestionReport,
        request: str,
        configuration: dict[str, Any],
    ) -> IndexOperation:
        assert report.repository is not None
        native = report.repository.github_id
        existing = session.scalars(
            select(Repository).where(Repository.github_id == native)
        ).one_or_none()
        if existing is not None and existing.name != name:
            raise ValueError("Repository rename requires explicit metadata migration")
        named = session.scalars(select(Repository).where(Repository.name == name)).one_or_none()
        if named is not None and named.github_id not in {None, native}:
            raise ValueError("Repository name now belongs to a different native identity")
        repo = register_repository(
            session,
            name,
            github_id=native,
            default_branch=report.repository.default_branch,
            visibility=report.repository.visibility,
        )
        operation = session.scalars(
            select(IndexOperation).where(
                IndexOperation.repository_id == repo.id,
                IndexOperation.request_key == request,
            )
        ).one_or_none()
        head = session.get(IndexHead, repo.id)
        if operation is None:
            if head:
                current = session.get(IndexOperation, head.revision_id)
                assert current is not None
                prior_report = IngestionReport.model_validate(current.scan_report)
                if report.started_at < prior_report.started_at:
                    raise ValueError("Snapshot predates the published scan")
            # Only one pending plan per repository; require replay before advancing its base.
            pending = session.scalars(
                select(IndexOperation).where(
                    IndexOperation.repository_id == repo.id,
                    IndexOperation.status.not_in(["READY", "RETIRED"]),
                )
            ).first()
            if pending is not None:
                raise IndexingError(
                    f"Replay pending revision {pending.id} before a new snapshot",
                    code="repository_busy",
                    retryable=True,
                )
            operation = IndexOperation(
                repository_id=repo.id,
                request_key=request,
                base_revision=head.revision_id if head else None,
                status="PREPARING",
                configuration=configuration,
                scan_report=report.model_dump(mode="json"),
            )
            session.add(operation)
        elif operation.configuration != configuration:
            raise ValueError("Operation configuration mismatch")
        elif operation.scan_report != report.model_dump(mode="json"):
            raise ValueError("Scan report changed for an existing operation")
        if operation.status in {"RETIRING", "RETIRED"}:
            raise IndexingError("Retired operation cannot be replayed; use a new scan")
        if operation.status != "READY":
            if operation.base_revision != (head.revision_id if head else None):
                raise IndexingError("Operation base revision is no longer current")
            operation.attempts = (operation.attempts or 0) + 1
            operation.error_code = None
            operation.failure_stage = None
        session.commit()
        return operation

    def _execute(self, session: Session, operation: IndexOperation) -> None:
        assert operation.plan is not None
        manifests = list(
            session.scalars(
                select(ChunkManifest)
                .where(ChunkManifest.operation_id == operation.id)
                .order_by(ChunkManifest.chunk_id)
            )
        )
        chunks = [DocumentChunk.model_validate(m.payload) for m in manifests]
        ids = [c.chunk_id for c in chunks]
        operation.status = "VECTOR_WRITES"
        session.commit()
        self.boundary("vector_writes_started")
        self.store.ensure(self.encoder.dimension)
        present = self.store.lookup(ids)
        missing = [c for c in chunks if c.chunk_id not in present]
        # Reuse same-model vectors for identical text even if metadata or ordinal changed.
        previous = (
            session.get(IndexOperation, operation.base_revision)
            if operation.base_revision
            else None
        )
        reuse: dict[str, list[float]] = {}
        if (
            missing
            and previous
            and previous.configuration["embedding_version"] == self.embedding_version
        ):
            old_ids = list(
                session.scalars(
                    select(ChunkManifest.chunk_id).where(ChunkManifest.operation_id == previous.id)
                )
            )
            reuse = {chunk.text: vector for chunk, vector in self.store.lookup(old_ids).values()}
        needed = list(dict.fromkeys(c.text for c in missing if c.text not in reuse))
        if needed:
            vectors = self.encoder.encode(needed)
            if len(vectors) != len(needed):
                raise ValueError("Encoder cardinality mismatch")
            reuse.update(zip(needed, vectors, strict=True))
        if missing:
            self.boundary("before_vector_writes")
            self.store.upsert(missing, [reuse[c.text] for c in missing], self.encoder.dimension)
        self.boundary("vectors_written")
        operation.status = "LEXICAL_BUILD"
        session.commit()
        path = self.root / str(operation.repository_id) / str(operation.id) / "chunks.jsonl"
        operation.lexical_sha256 = publish_lexical(path, chunks)
        session.commit()
        self.boundary("lexical_built")
        operation.status = "VERIFYING"
        session.commit()
        stored = self.store.lookup(ids)
        if set(stored) != set(ids) or any(stored[c.chunk_id][0] != c for c in chunks):
            raise ValueError("Vector manifest verification failed")
        lexical = [
            DocumentChunk.model_validate_json(line) for line in path.read_text().splitlines()
        ]
        if lexical != chunks or file_hash(path) != operation.lexical_sha256:
            raise ValueError("Lexical manifest verification failed")
        self.boundary("verified")
        if self.publication_guard is not None:
            self.publication_guard(session, operation.id)
        # Metadata and publication pointer share one PostgreSQL transaction.
        repo = session.scalars(
            select(Repository).where(Repository.id == operation.repository_id).with_for_update()
        ).one()
        if metadata_state(session, repo.id) != operation.plan["metadata_state"]:
            raise ValueError("Concurrent metadata import changed the prepared base")
        for data in operation.plan["documents"].values():
            doc = GitHubDocument.model_validate(data["payload"])
            upsert_document(session, repo, doc)
            record = session.scalars(
                select(Document).where(
                    Document.repository_id == repo.id,
                    Document.source_type == doc.document_type.value,
                    Document.source_key == source_key(doc),
                )
            ).one()
            record.index_status = "indexed"
            record.embedding_version, record.chunker_version = (
                self.embedding_version,
                self.chunker_version,
            )
        for key in operation.plan["deleted"]:
            family, native_key = json.loads(key)
            deleted_record = session.scalars(
                select(Document).where(
                    Document.repository_id == repo.id,
                    Document.source_type == family,
                    Document.source_key == native_key,
                )
            ).one_or_none()
            if deleted_record:
                deleted_record.index_status = "deleted"
                deleted_record.deleted_at = datetime.now(UTC)
        head = session.get(IndexHead, repo.id)
        if head is None:
            session.add(IndexHead(repository_id=repo.id, revision_id=operation.id))
        else:
            if head.revision_id != operation.base_revision:
                raise ValueError("Publication base changed")
            head.revision_id = operation.id
        operation.status, operation.error_code = "READY", None
        self.boundary("publishing")
        session.commit()

    def retire_revision(self, revision: UUID, *, repository: str) -> None:
        """Explicitly retire a non-current revision after its readers have drained.

        Retain audit manifests, remove only points not referenced by any other live
        operation. RETIRING is committed before deletion, so interrupted GC is replayable.
        """
        name = canonical_repository(repository)
        lock = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)
        with self.engine.connect() as connection:
            locked: bool = connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": lock}
            ).scalar_one()
            connection.commit()
            if not locked:
                raise IndexingError(
                    "Another indexing operation holds this repository lock",
                    code="repository_busy",
                    retryable=True,
                )
            try:
                with Session(connection, expire_on_commit=False) as session:
                    operation = session.scalars(
                        select(IndexOperation)
                        .join(Repository)
                        .where(
                            IndexOperation.id == revision,
                            Repository.name == name,
                        )
                    ).one()
                    if operation.configuration != {
                        **self.configuration,
                        "reconcile": operation.configuration["reconcile"],
                    }:
                        raise IndexingError("Retirement configuration mismatch")
                    head = session.get(IndexHead, operation.repository_id)
                    if head and head.revision_id == revision:
                        raise IndexingError("Cannot retire the current READY revision")
                    unfinished_job = session.scalars(
                        select(IngestionJob.id).where(
                            IngestionJob.result_revision_id == revision,
                            IngestionJob.status != "COMPLETE",
                        )
                    ).first()
                    if unfinished_job is not None:
                        raise IndexingError("Revision is still required by an unfinished job")
                    if operation.status == "RETIRED":
                        return
                    operation.status = "RETIRING"
                    session.commit()
                    candidates = set(
                        session.scalars(
                            select(ChunkManifest.chunk_id).where(
                                ChunkManifest.operation_id == revision
                            )
                        )
                    )
                    protected = set(
                        session.scalars(
                            select(ChunkManifest.chunk_id)
                            .join(IndexOperation, IndexOperation.id == ChunkManifest.operation_id)
                            .where(
                                IndexOperation.id != revision, IndexOperation.status != "RETIRED"
                            )
                        )
                    )
                    stale = sorted(candidates - protected)
                    for start in range(0, len(stale), self.store.batch_size):
                        self.store.delete(stale[start : start + self.store.batch_size])
                        self.boundary("retirement_delete")
                    path = self.root / str(operation.repository_id) / str(revision) / "chunks.jsonl"
                    path.unlink(missing_ok=True)
                    operation.status = "RETIRED"
                    session.commit()
            except IndexingError:
                raise
            except Exception:
                raise IndexingError(
                    f"Retirement of revision {revision} is incomplete; retry retirement. "
                    "Current READY is preserved."
                ) from None
            finally:
                connection.rollback()
                connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock})
                connection.commit()

    def ready(self, repository: str) -> "ReadyIndex":
        with Session(self.engine) as session:
            operation = session.scalars(
                select(IndexOperation)
                .join(IndexHead, IndexHead.revision_id == IndexOperation.id)
                .join(Repository, Repository.id == IndexHead.repository_id)
                .where(
                    Repository.name == canonical_repository(repository),
                    IndexOperation.status == "READY",
                )
            ).one_or_none()
            if operation is None:
                raise IndexingError("Repository has no READY indexing revision")
            if any(
                operation.configuration[k] != self.configuration[k]
                for k in (
                    "root",
                    "qdrant_path",
                    "collection",
                    "embedding_version",
                    "chunker_version",
                )
            ):
                raise IndexingError("Reader configuration differs from READY revision")
            manifests = list(
                session.scalars(
                    select(ChunkManifest).where(ChunkManifest.operation_id == operation.id)
                )
            )
            path = self.root / str(operation.repository_id) / str(operation.id) / "chunks.jsonl"
            if not path.exists() or file_hash(path) != operation.lexical_sha256:
                raise IndexingError("READY lexical snapshot unavailable or corrupt")
            return ReadyIndex(operation.id, path, [m.chunk_id for m in manifests], self.store)


class ReadyIndex:
    """Pin an immutable revision for both backends; never search the raw application collection."""

    def __init__(
        self, revision: UUID, path: Path, ids: list[str], store: IncrementalVectorStore
    ) -> None:
        self.revision, self.ids, self.store = revision, ids, store
        self.lexical = BM25Retriever(path)
        self.chunks = [
            DocumentChunk.model_validate_json(line) for line in path.read_text().splitlines()
        ]

    def vector_search(self, vector: list[float], *, top_k: int = 10) -> list[VectorMatch]:
        stored = self.store.lookup(self.ids)
        if set(stored) != set(self.ids) or any(
            stored[chunk.chunk_id][0] != chunk for chunk in self.chunks
        ):
            raise IndexingError("READY vectors unavailable; refusing incomplete retrieval")
        return self.store.search_membership(vector, self.ids, top_k)
