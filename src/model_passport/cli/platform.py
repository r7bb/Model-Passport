"""Platform operator commands: keys, migrations, the first super admin, the API, and workers.

Settings come from ``MP_*`` environment variables (see ``model_passport.platform.settings``).
"""

# ruff: noqa: PLC0415 - the platform stack is imported only when these commands run

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated

import typer

from model_passport.cli._app import SHARE, app, fail

platform_app = typer.Typer(help="Run the multi-tenant MP platform.", no_args_is_help=True)
if TYPE_CHECKING:
    from model_passport.platform.settings import Settings

app.add_typer(platform_app, name="platform", rich_help_panel=SHARE)


def _settings() -> Settings:
    from model_passport.platform.settings import Settings, SettingsError

    try:
        return Settings.from_env()
    except SettingsError as exc:
        fail(f"{exc} (run `passport platform keygen` for values)")


@platform_app.command()
def keygen() -> None:
    """Print fresh random values for MP_MASTER_KEY and MP_JWT_SECRET."""
    import secrets

    from model_passport.platform.settings import new_key

    typer.echo(f"MP_MASTER_KEY={new_key()}")
    typer.echo(f"MP_JWT_SECRET={secrets.token_urlsafe(48)}")


@platform_app.command()
def migrate() -> None:
    """Create or upgrade the database schema."""
    from model_passport.platform.db import make_engine
    from model_passport.platform.db import migrate as run_migrations

    settings = _settings()
    run_migrations(make_engine(settings.database_url))
    typer.echo("database is up to date")


@platform_app.command("create-superadmin")
def create_superadmin(
    email: Annotated[str, typer.Option(help="Email address to sign in with.")],
    password: Annotated[str, typer.Option(prompt=True, hide_input=True, confirmation_prompt=True)],
    name: str = "",
) -> None:
    """Create the platform super admin, who creates organizations."""
    from model_passport.platform import services
    from model_passport.platform.db import ALL_TENANTS, make_engine, scoped_session, session_factory

    settings = _settings()
    try:
        with scoped_session(session_factory(make_engine(settings.database_url)), ALL_TENANTS) as s:
            services.create_user(s, email, password, name, super_admin=True)
    except services.ServiceError as exc:
        fail(str(exc))
    typer.echo(f"created super admin {email}")


@platform_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8080,
) -> None:
    """Run the platform API (migrations run on start)."""
    import uvicorn

    from model_passport.platform.api.app import create_app

    uvicorn.run(create_app(_settings()), host=host, port=port)


@platform_app.command()
def worker(
    once: Annotated[bool, typer.Option("--once", help="Drain the queue, then exit.")] = False,
    idle: Annotated[float, typer.Option(help="Seconds to wait when the queue is empty.")] = 2.0,
) -> None:
    """Run a worker that processes train, audit, remediate, and report jobs."""
    from model_passport.platform.db import make_engine, session_factory
    from model_passport.platform.storage import open_store
    from model_passport.platform.worker import WorkerContext, drain, run_once

    settings = _settings()
    context = WorkerContext(
        factory=session_factory(make_engine(settings.database_url)),
        store=open_store(settings.storage, settings.s3_endpoint),
        master_key=settings.master_key,
    )
    if once:
        typer.echo(f"processed {drain(context)} job(s)")
        return
    typer.echo(f"worker {context.name} waiting for jobs (Ctrl+C to stop)")
    while True:
        if not run_once(context):
            time.sleep(idle)
