"""Persist canonical membership before any external writes; unchanged documents reuse manifests."""

import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from github_intelligence.indexing.models import ChunkManifest, IndexOperation
from github_intelligence.indexing.policy import chunk_identity, deletion_reason, digest
from github_intelligence.ingestion.models import IngestionReport
from github_intelligence.models import GitHubDocument
from github_intelligence.persistence.identity import (
    content_hash,
    document_id,
    normalized_payload,
    source_key,
)
from github_intelligence.persistence.models import Document
from github_intelligence.retrieval import DocumentChunker


def identity_key(family: str, key: str) -> str:
    return json.dumps([family, key], separators=(",", ":"))


def metadata_state(session: Session, repository_id: UUID) -> str:
    documents = session.scalars(select(Document).where(Document.repository_id == repository_id))
    return digest(
        sorted(
            (
                str(d.id),
                d.content_hash,
                d.index_status,
                str(d.deleted_at),
                d.embedding_version,
                d.chunker_version,
            )
            for d in documents
        )
    )


def prepare_plan(
    session: Session,
    operation: IndexOperation,
    incoming: list[GitHubDocument],
    report: IngestionReport,
    reconcile: bool,
    *,
    chunker: DocumentChunker,
    embedding_version: str,
    chunker_version: str,
    boundary: Callable[[str], None],
    explicit_deletions: set[str] | None = None,
) -> None:
    if operation.plan is not None:
        return
    previous = (
        session.get(IndexOperation, operation.base_revision) if operation.base_revision else None
    )
    if previous and any(
        previous.configuration[k] != operation.configuration[k]
        for k in ("root", "qdrant_path", "collection")
    ):
        raise ValueError("Moving published storage requires an explicit migration")
    prior_docs: dict[str, Any] = (
        dict(previous.plan["documents"]) if previous and previous.plan else {}
    )
    prior_chunks = (
        list(
            session.scalars(
                select(ChunkManifest).where(ChunkManifest.operation_id == operation.base_revision)
            )
        )
        if previous
        else []
    )
    by_document: dict[str, list[ChunkManifest]] = {}
    for chunk in prior_chunks:
        by_document.setdefault(identity_key(chunk.source_type, chunk.source_key), []).append(chunk)
    records = list(
        session.scalars(select(Document).where(Document.repository_id == operation.repository_id))
    )
    records_by_key = {identity_key(d.source_type, d.source_key): d for d in records}
    active: dict[str, Any] = dict(prior_docs)
    # Previously imported metadata is canonical input too; absence is never automatic deletion.
    for metadata_record in records:
        if metadata_record.deleted_at is None:
            active[identity_key(metadata_record.source_type, metadata_record.source_key)] = {
                "payload": metadata_record.normalized_payload,
                "scope": prior_docs.get(
                    identity_key(metadata_record.source_type, metadata_record.source_key), {}
                ).get("scope", {}),
            }
    observed = {identity_key(d.document_type.value, source_key(d)): d for d in incoming}
    counts = {"new": 0, "changed": 0, "unchanged": 0, "deleted": 0, "skipped": 0}
    reasons: dict[str, str] = {}
    deleted: list[str] = []
    for key, doc in observed.items():
        record = records_by_key.get(key)
        if record is None:
            counts["new"] += 1
        elif record.content_hash != content_hash(doc) or record.deleted_at is not None:
            counts["changed"] += 1
        else:
            counts["unchanged"] += 1
        active[key] = {
            "payload": normalized_payload(doc),
            "scope": report.sources[doc.document_type.value].scope,
        }
    for key in list(active):
        if key in observed:
            continue
        family, native_key = json.loads(key)
        reason = (
            None
            if key in (explicit_deletions or set())
            else deletion_reason(
                report, family, active[key]["scope"], native_key, reconcile=reconcile
            )
        )
        if reason is None:
            del active[key]
            deleted.append(key)
            counts["deleted"] += 1
        else:
            reasons[key] = reason
            counts["skipped"] += 1
    manifests: list[ChunkManifest] = []
    for key, data in sorted(active.items()):
        doc = GitHubDocument.model_validate(data["payload"])
        old = prior_docs.get(key)
        reusable = (
            old is not None
            and old["payload"] == data["payload"]
            and previous is not None
            and previous.configuration["embedding_version"] == embedding_version
            and previous.configuration["chunker_version"] == chunker_version
        )
        if reusable:
            for chunk in by_document.get(key, []):
                manifests.append(
                    ChunkManifest(
                        operation_id=operation.id,
                        chunk_id=chunk.chunk_id,
                        document_identity=chunk.document_identity,
                        source_type=chunk.source_type,
                        source_key=chunk.source_key,
                        ordinal=chunk.ordinal,
                        chunker_version=chunk.chunker_version,
                        embedding_version=chunk.embedding_version,
                        payload_hash=chunk.payload_hash,
                        payload=chunk.payload,
                    )
                )
            continue
        family, native_key = json.loads(key)
        record = records_by_key.get(key)
        doc_id = record.id if record else document_id(doc)
        for new_chunk in chunker.chunk(doc):
            payload = new_chunk.model_dump(mode="json")
            payload.pop("chunk_id")
            payload["document_id"] = str(doc_id)
            payload_hash = digest(payload)
            point_id = chunk_identity(
                operation.repository_id,
                family,
                native_key,
                new_chunk.chunk_index,
                payload_hash,
                embedding_version,
                chunker_version,
            )
            payload["chunk_id"] = point_id
            manifests.append(
                ChunkManifest(
                    operation_id=operation.id,
                    chunk_id=point_id,
                    document_identity=doc_id,
                    source_type=family,
                    source_key=native_key,
                    ordinal=new_chunk.chunk_index,
                    chunker_version=chunker_version,
                    embedding_version=embedding_version,
                    payload_hash=payload_hash,
                    payload=payload,
                )
            )
    operation.plan = {
        "documents": active,
        "deleted": deleted,
        "counts": counts,
        "preserved": reasons,
        "metadata_state": metadata_state(session, operation.repository_id),
    }
    session.add_all(manifests)
    operation.status = "VECTOR_WRITES"
    session.commit()
    boundary("prepared")
