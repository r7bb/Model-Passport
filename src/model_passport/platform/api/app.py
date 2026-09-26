"""The platform API application (``passport platform serve``)."""

from __future__ import annotations

from fastapi import FastAPI

from model_passport import __version__
from model_passport.platform.api import routes_admin, routes_tenant
from model_passport.platform.api.deps import AppState
from model_passport.platform.db import make_engine, migrate, session_factory
from model_passport.platform.settings import Settings
from model_passport.platform.storage import open_store

API_PREFIX = "/api/v1"


def create_app(settings: Settings, run_migrations: bool = True) -> FastAPI:
    engine = make_engine(settings.database_url)
    if run_migrations:
        migrate(engine)
    app = FastAPI(
        title="MP platform API",
        version=__version__,
        description="Entity-level PII leakage auditing and remediation. Tenant routes need the "
        "organization's subdomain or an X-MP-Tenant header.",
    )
    app.state.mp = AppState(
        settings=settings,
        factory=session_factory(engine),
        store=open_store(settings.storage, settings.s3_endpoint),
    )
    app.include_router(routes_admin.router, prefix=API_PREFIX)
    app.include_router(routes_tenant.router, prefix=API_PREFIX)

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app
