"""Application index revisions, explicit membership and publication pointer."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0003_incremental_index"
down_revision = "0002_review_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_operations",
        sa.Column("id", pg.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("repository_id", pg.UUID(), sa.ForeignKey("repositories.id"), nullable=False),
        sa.Column("request_key", sa.Text(), nullable=False),
        sa.Column("base_revision", pg.UUID(), sa.ForeignKey("index_operations.id")),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("configuration", pg.JSONB(), nullable=False),
        sa.Column("scan_report", pg.JSONB(), nullable=False),
        sa.Column("plan", pg.JSONB()),
        sa.Column("lexical_sha256", sa.Text()),
        sa.Column("error_code", sa.Text()),
        sa.Column("failure_stage", sa.String(20)),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint("repository_id", "request_key", name="uq_index_request"),
        sa.CheckConstraint(
            "status IN ('PREPARING','VECTOR_WRITES','LEXICAL_BUILD','VERIFYING',"
            "'READY','FAILED','RETIRING','RETIRED')",
            name="ck_index_status",
        ),
    )
    op.create_index("ix_index_operations_repository_id", "index_operations", ["repository_id"])
    op.create_table(
        "chunk_manifests",
        sa.Column(
            "operation_id", pg.UUID(), sa.ForeignKey("index_operations.id"), primary_key=True
        ),
        sa.Column("chunk_id", sa.Text(), primary_key=True),
        sa.Column("document_identity", pg.UUID(), nullable=False),
        sa.Column("source_type", sa.String(40), nullable=False),
        sa.Column("source_key", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("chunker_version", sa.Text(), nullable=False),
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column("payload_hash", sa.Text(), nullable=False),
        sa.Column("payload", pg.JSONB(), nullable=False),
    )
    op.create_index(
        "ix_chunk_manifests_document_identity", "chunk_manifests", ["document_identity"]
    )
    op.create_table(
        "index_heads",
        sa.Column("repository_id", pg.UUID(), sa.ForeignKey("repositories.id"), primary_key=True),
        sa.Column(
            "revision_id",
            pg.UUID(),
            sa.ForeignKey("index_operations.id"),
            nullable=False,
            unique=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("index_heads")
    op.drop_table("chunk_manifests")
    op.drop_table("index_operations")
