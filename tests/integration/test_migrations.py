import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("alembic")

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from github_intelligence.persistence.models import Base


def test_upgrade_downgrade_reupgrade_and_model_alignment(postgres_engine, migration_config) -> None:
    with postgres_engine.begin() as conn:
        migration_config.attributes["connection"] = conn
        command.upgrade(migration_config, "head")
        assert set(inspect(conn).get_table_names()) == set(Base.metadata.tables) | {
            "alembic_version"
        }
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
        # Exercise data and server UUID/timestamp defaults, not just table existence.
        conn.execute(text("INSERT INTO users (external_identity) VALUES ('test-subject')"))
        row = conn.execute(text("SELECT id, created_at FROM users")).one()
        assert row.id is not None and row.created_at.tzinfo is not None
        command.downgrade(migration_config, "base")
        assert inspect(conn).get_table_names() == ["alembic_version"]
        command.upgrade(migration_config, "head")
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
        assert conn.execute(text("SELECT count(*) FROM users")).scalar_one() == 0


def test_environment_driven_alembic_cli_flow(postgres_engine) -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["DATABASE_URL"] = postgres_engine.url.render_as_string(hide_password=False)
    for arguments in [("upgrade", "head"), ("downgrade", "base"), ("upgrade", "head"), ("check",)]:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(root / "alembic.ini"), *arguments],
            env=env,
            capture_output=True,
            text=True,
            cwd=root,
        )
        assert result.returncode == 0, result.stderr
    with postgres_engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == set(Base.metadata.tables) | {
            "alembic_version"
        }


def test_review_constraint_downgrade_preserves_data_on_refusal(
    postgres_engine, migration_config
) -> None:
    from sqlalchemy.exc import IntegrityError

    from github_intelligence.models import GitHubDocument
    from github_intelligence.persistence.database import session_factory
    from github_intelligence.persistence.operations import import_documents

    with postgres_engine.begin() as conn:
        migration_config.attributes["connection"] = conn
        command.upgrade(migration_config, "head")
    with session_factory(postgres_engine).begin() as session:
        import_documents(
            session,
            [
                GitHubDocument(
                    repository="acme/repo",
                    document_type="pull_request_review",
                    github_id="23",
                    title="Review",
                    text="Review summary",
                    source_url="https://github.com/acme/repo/pull/1#pullrequestreview-23",
                )
            ],
        )
    with pytest.raises(IntegrityError), postgres_engine.begin() as conn:
        migration_config.attributes["connection"] = conn
        command.downgrade(migration_config, "0001_application_state")
    with postgres_engine.begin() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM documents WHERE source_type='pull_request_review'")
            ).scalar_one()
            == 1
        )
        assert (
            conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == "0005_webhook_deliveries"
        )
