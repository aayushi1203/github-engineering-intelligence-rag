"""Durable claims, retry scheduling and READY result for the existing job table."""

from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0004_job_leases"
down_revision = "0003_incremental_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns: list[tuple[str, sa.types.TypeEngine[Any]]] = [
        ("lease_owner", UUID()),
        ("lease_expires_at", sa.DateTime(timezone=True)),
        ("heartbeat_at", sa.DateTime(timezone=True)),
        ("next_attempt_at", sa.DateTime(timezone=True)),
        ("snapshot_path", sa.Text()),
        ("index_stage", sa.Text()),
    ]
    for name, datatype in columns:
        op.add_column("ingestion_jobs", sa.Column(name, datatype))
    op.add_column(
        "ingestion_jobs",
        sa.Column("retryable", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("ingestion_jobs", sa.Column("result_revision_id", UUID()))
    op.create_foreign_key(
        "fk_job_result_revision",
        "ingestion_jobs",
        "index_operations",
        ["result_revision_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_job_result_revision", "ingestion_jobs", type_="foreignkey")
    for name in [
        "result_revision_id",
        "retryable",
        "lease_owner",
        "lease_expires_at",
        "heartbeat_at",
        "next_attempt_at",
        "snapshot_path",
        "index_stage",
    ]:
        op.drop_column("ingestion_jobs", name)
