"""Explicit, PostgreSQL-only engine and caller-owned transaction boundaries."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


class PersistenceError(RuntimeError):
    """Public error containing no connection string or SQL parameters."""


def create_database_engine(database_url: str | None) -> Engine:
    if not database_url:
        raise PersistenceError("DATABASE_URL is required for metadata persistence")
    try:
        url = make_url(database_url)
        if url.drivername not in {"postgresql", "postgresql+psycopg"}:
            raise ValueError("Unsupported database driver")
        url = url.set(drivername="postgresql+psycopg")
        options = str(url.query.get("options", ""))
        return create_engine(
            url,
            pool_pre_ping=True,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 10,
                "options": options + " -c timezone=UTC -c statement_timeout=30000",
            },
        )
    except (ValueError, TypeError) as error:
        raise PersistenceError("DATABASE_URL must be a valid PostgreSQL psycopg URL") from error


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
