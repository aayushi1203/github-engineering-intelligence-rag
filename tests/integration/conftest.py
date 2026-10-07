"""Real PostgreSQL tests isolated in disposable, randomly named schemas."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from sqlalchemy import Engine


@pytest.fixture
def postgres_engine() -> Iterator["Engine"]:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg")
    from sqlalchemy import text
    from sqlalchemy.engine import make_url

    from github_intelligence.persistence.database import create_database_engine

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run real PostgreSQL integration tests")
    parsed = make_url(url)
    if not parsed.database or "test" not in parsed.database.lower():
        pytest.fail("TEST_DATABASE_URL must name an explicitly disposable test database")
    schema = "phase1_test_" + uuid4().hex
    admin = create_database_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped_url = parsed.update_query_dict({"options": f"-c search_path={schema}"})
    engine = create_database_engine(scoped_url.render_as_string(hide_password=False))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def migration_config(postgres_engine: "Engine") -> object:
    pytest.importorskip("alembic")
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]
    return Config(str(root / "alembic.ini"))


@pytest.fixture
def migrated_engine(postgres_engine: "Engine", migration_config: object) -> "Engine":
    from alembic import command

    with postgres_engine.begin() as conn:
        migration_config.attributes["connection"] = conn
        command.upgrade(migration_config, "head")
    migration_config.attributes.pop("connection", None)
    return postgres_engine
