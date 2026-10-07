"""Bounded startup probes for real PostgreSQL and Qdrant CI services."""

import os
import time
import urllib.request

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def main() -> None:
    engine = create_engine(os.environ["TEST_DATABASE_URL"], connect_args={"connect_timeout": 2})
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            try:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                with urllib.request.urlopen(
                    os.environ["TEST_QDRANT_URL"] + "/readyz", timeout=2
                ) as response:
                    if response.status == 200:
                        print("PostgreSQL and Qdrant are ready")
                        return
            except (OSError, SQLAlchemyError):
                pass  # Retry without exposing connection strings or passwords.
            time.sleep(1)
        raise TimeoutError("CI storage services did not become ready within 60 seconds")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
