"""The root Typer app and helpers shared by command modules."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
import yaml
from pydantic import ValidationError

from model_passport.core.config import CONFIG_FILENAME, ProjectConfig, load_config

app = typer.Typer(
    help="Signed, verifiable passports for trained ML models.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

# Exit codes: 0 ok / pass, 1 policy fail / verification fail / drift, 2 usage or input error.
EXIT_FAIL = 1
EXIT_ERROR = 2

ConfigOption = Annotated[Path, typer.Option(help="Project config file.")]
DEFAULT_CONFIG = Path(CONFIG_FILENAME)


def warn(message: str) -> None:
    typer.echo(f"warning: {message}", err=True)


def fail(message: str, code: int = EXIT_ERROR) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def load_project(config: Path) -> ProjectConfig:
    try:
        return load_config(config)
    except (OSError, ValidationError, yaml.YAMLError) as exc:
        fail(f"cannot load {config}: {exc}")
