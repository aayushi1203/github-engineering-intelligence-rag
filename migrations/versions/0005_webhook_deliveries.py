"""Authenticated, deduplicated GitHub receipts linked to the existing durable queue."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0005_webhook_deliveries"
down_revision = "0004_job_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "webhook_deliveries",
        sa.Column("delivery_id", sa.String(128), primary_key=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("action", sa.String(64)),
        sa.Column("repository_id", UUID(), sa.ForeignKey("repositories.id")),
        sa.Column("repository_github_id", sa.Text()),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("job_id", UUID(), sa.ForeignKey("ingestion_jobs.id")),
        sa.Column("body_sha256", sa.String(64), nullable=False),
        sa.Column("event_metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("error_code", sa.String(64)),
        sa.CheckConstraint(
            "status IN ('ACCEPTED','PROCESSING','COMPLETE','FAILED','IGNORED','DEFERRED')",
            name="ck_webhook_status",
        ),
    )
    op.create_index("ix_webhook_deliveries_repository_id", "webhook_deliveries", ["repository_id"])
    op.create_index("ix_webhook_deliveries_job_id", "webhook_deliveries", ["job_id"])


def downgrade() -> None:
    op.drop_table("webhook_deliveries")
