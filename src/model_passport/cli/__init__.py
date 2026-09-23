"""``passport`` command line interface. Command modules register themselves on ``app``."""

from model_passport.cli import ops, project, scan  # noqa: F401  (registers commands)
from model_passport.cli._app import app

__all__ = ["app"]
