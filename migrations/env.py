"""Environment-driven PostgreSQL migrations; no implicit table creation."""

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.exc import SQLAlchemyError

from github_intelligence.config import Settings
from github_intelligence.indexing import models as index_models  # noqa: F401
from github_intelligence.persistence.database import PersistenceError, create_database_engine
from github_intelligence.persistence.models import Base

config = context.config
# Tests may supply an existing connection to their isolated schema.
connection = config.attributes.get("connection")


def run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if connection is not None:
    run(connection)
else:
    try:
        engine = create_database_engine(Settings().database_url)
        try:
            if context.is_offline_mode():
                context.configure(
                    url=engine.url,
                    target_metadata=Base.metadata,
                    literal_binds=True,
                    dialect_opts={"paramstyle": "named"},
                )
                with context.begin_transaction():
                    context.run_migrations()
            else:
                with engine.connect() as active_connection:
                    run(active_connection)
        finally:
            engine.dispose()
    except (SQLAlchemyError, PersistenceError):
        raise PersistenceError(
            "Migration failed; verify DATABASE_URL and PostgreSQL access"
        ) from None
