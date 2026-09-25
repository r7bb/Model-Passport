"""Database engine, sessions, and migrations.

On PostgreSQL every transaction sets ``app.tenant_id``; the row-level security policies from
the initial migration only let it see that tenant's rows (``*`` for platform-wide work by
super admins and workers). A query that forgets its tenant filter therefore still cannot read
another tenant's data. SQLite (local use and tests) relies on the application-level scoping.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

ALL_TENANTS = "*"


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(connection: Any, _record: Any) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_engine(url, pool_pre_ping=True)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def scoped_session(factory: sessionmaker[Session], tenant_id: str) -> Iterator[Session]:
    """A session whose every transaction is limited to ``tenant_id`` (or ``*``)."""
    session = factory()

    @event.listens_for(session, "after_begin")
    def _set_tenant(_session: Session, _transaction: Any, connection: Any) -> None:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": tenant_id}
            )

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def alembic_config(engine: Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("model_passport.platform") / "migrations"))
    config.attributes["engine"] = engine
    return config


def migrate(engine: Engine, revision: str = "head") -> None:
    """Bring the schema up to ``revision`` (the latest by default)."""
    command.upgrade(alembic_config(engine), revision)
