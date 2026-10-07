"""Allow distinct PR review summary documents without changing legacy IDs."""

from alembic import op

revision = "0002_review_source"
down_revision = "0001_application_state"
branch_labels = None
depends_on = None

_OLD = (
    "source_type IN ('issue','issue_comment','pull_request','pull_request_comment',"
    "'pull_request_review_comment','commit','release','documentation')"
)
_NEW = (
    "source_type IN ('issue','issue_comment','pull_request','pull_request_comment',"
    "'pull_request_review_comment','pull_request_review','commit','release','documentation')"
)


def upgrade() -> None:
    op.drop_constraint("ck_document_source_type", "documents", type_="check")
    op.create_check_constraint("ck_document_source_type", "documents", _NEW)


def downgrade() -> None:
    # PostgreSQL refuses this constraint if review rows remain. Never silently delete them.
    op.drop_constraint("ck_document_source_type", "documents", type_="check")
    op.create_check_constraint("ck_document_source_type", "documents", _OLD)
