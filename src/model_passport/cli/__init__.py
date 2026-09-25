"""``passport`` command line interface. Command modules register themselves on ``app``."""

# Import order sets the order of the help panels: project, ops, then scan.
from model_passport.cli import project  # isort: skip
from model_passport.cli import ops  # isort: skip
from model_passport.cli import scan  # isort: skip
from model_passport.cli._app import app

__all__ = ["app", "ops", "project", "scan"]
