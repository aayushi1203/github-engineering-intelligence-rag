"""Revision manifests and the single PostgreSQL publication pointer."""

from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from github_intelligence.persistence.models import Base, Identity


class IndexOperation(Identity, Base):
    __tablename__ = "index_operations"
    __table_args__ = (
        UniqueConstraint("repository_id", "request_key", name="uq_index_request"),
        CheckConstraint(
            "status IN ('PREPARING','VECTOR_WRITES','LEXICAL_BUILD','VERIFYING',"
            "'READY','FAILED','RETIRING','RETIRED')",
            name="ck_index_status",
        ),
    )
    repository_id: Mapped[UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    request_key: Mapped[str] = mapped_column(Text)
    base_revision: Mapped[UUID | None] = mapped_column(ForeignKey("index_operations.id"))
    status: Mapped[str] = mapped_column(String(20))
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB)
    scan_report: Mapped[dict[str, Any]] = mapped_column(JSONB)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    lexical_sha256: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(Text)
    failure_stage: Mapped[str | None] = mapped_column(String(20))
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")


class ChunkManifest(Base):
    __tablename__ = "chunk_manifests"
    operation_id: Mapped[UUID] = mapped_column(ForeignKey("index_operations.id"), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_identity: Mapped[UUID] = mapped_column(index=True)
    source_type: Mapped[str] = mapped_column(String(40))
    source_key: Mapped[str] = mapped_column(Text)
    ordinal: Mapped[int] = mapped_column(Integer)
    chunker_version: Mapped[str] = mapped_column(Text)
    embedding_version: Mapped[str] = mapped_column(Text)
    payload_hash: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class IndexHead(Base):
    __tablename__ = "index_heads"
    repository_id: Mapped[UUID] = mapped_column(ForeignKey("repositories.id"), primary_key=True)
    revision_id: Mapped[UUID] = mapped_column(ForeignKey("index_operations.id"), unique=True)
