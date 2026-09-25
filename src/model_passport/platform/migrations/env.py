"""Alembic environment: runs migrations on the engine passed in by ``db.migrate``."""

from alembic import context

from model_passport.platform.models import Base

config = context.config
engine = config.attributes["engine"]

with engine.begin() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    context.run_migrations()
