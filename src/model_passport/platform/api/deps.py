"""Request context: who is calling, for which tenant, with which role.

The tenant comes from the ``X-MP-Tenant`` header (set by the API gateway from the subdomain)
or from the ``Host`` header (``<slug>.<base domain>``). Membership is checked in the database
on every request, and the handler then runs in a session limited to that tenant, so row-level
security applies to everything it does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from model_passport.platform import auditlog, lifecycle, services
from model_passport.platform.api.schemas import MemberIn
from model_passport.platform.controlplane import ControlPlane
from model_passport.platform.db import ALL_TENANTS, scoped_session
from model_passport.platform.models import Membership, Role, Tenant, TenantStatus, User
from model_passport.platform.rbac import Permission, allowed
from model_passport.platform.security import TokenError, read_token
from model_passport.platform.settings import Settings
from model_passport.platform.storage import ObjectStore, TenantStore

bearer = HTTPBearer(auto_error=False)


@dataclass
class AppState:
    settings: Settings
    factory: sessionmaker[Session]
    store: ObjectStore


def state(request: Request) -> AppState:
    app_state: AppState = request.app.state.mp
    return app_state


State = Annotated[AppState, Depends(state)]


def _user(app: AppState, credentials: HTTPAuthorizationCredentials | None) -> User:
    if credentials is None:
        raise HTTPException(401, "sign in first", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = read_token(credentials.credentials, app.settings.jwt_secret)
    except TokenError as exc:
        raise HTTPException(401, "session expired or invalid; sign in again") from exc
    with scoped_session(app.factory, ALL_TENANTS) as session:
        user = session.get(User, claims["sub"])
        if user is None or user.disabled:
            raise HTTPException(401, "account not found or disabled")
        session.expunge(user)
    return user


def current_user(
    app: State, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> User:
    return _user(app, credentials)


CurrentUser = Annotated[User, Depends(current_user)]


def tenant_slug(settings: Settings, host: str | None, header: str | None) -> str | None:
    if header:
        return header.strip().lower()
    hostname = (host or "").split(":")[0].lower()
    suffix = "." + settings.base_domain
    return hostname[: -len(suffix)] if hostname.endswith(suffix) else None


@dataclass
class Ctx:
    """Everything a tenant handler needs, already authorized."""

    session: Session
    user: User
    tenant: Tenant
    role: Role | None
    files: TenantStore
    token: str
    controlplane_addr: str | None

    @property
    def actor(self) -> auditlog.Actor:
        return auditlog.Actor(self.user.id, self.user.email)

    @property
    def who(self) -> lifecycle.Who:
        return lifecycle.Who(self.actor, self.role, self.user.is_super_admin)

    def can(self, permission: Permission) -> bool:
        return allowed(self.role, permission, self.user.is_super_admin)

    @property
    def controlplane(self) -> ControlPlane | None:
        """The control plane acting as this caller, or None when none is configured."""
        if not self.controlplane_addr:
            return None
        return ControlPlane(self.controlplane_addr, self.token, self.tenant.slug)


def _membership(app: AppState, user: User, slug: str) -> tuple[Tenant, Role | None]:
    with scoped_session(app.factory, ALL_TENANTS) as session:
        tenant = session.scalars(select(Tenant).where(Tenant.slug == slug)).first()
        if tenant is None:
            raise HTTPException(404, f"organization {slug!r} not found")
        if tenant.status is TenantStatus.SUSPENDED and not user.is_super_admin:
            raise HTTPException(403, "this organization is suspended")
        member = session.scalars(
            select(Membership).where(
                Membership.tenant_id == tenant.id, Membership.user_id == user.id
            )
        ).first()
        if member is None and not user.is_super_admin:
            raise HTTPException(403, "you are not a member of this organization")
        session.expunge(tenant)
        return tenant, member.role if member else None


def require(permission: Permission) -> Callable[..., Iterator[Ctx]]:
    """A dependency that yields a tenant context if the caller has ``permission`` there."""

    def dependency(
        request: Request,
        app: State,
        user: CurrentUser,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
        x_mp_tenant: Annotated[str | None, Header()] = None,
    ) -> Iterator[Ctx]:
        slug = tenant_slug(app.settings, request.headers.get("host"), x_mp_tenant)
        if not slug:
            raise HTTPException(400, "no organization: use its subdomain or the X-MP-Tenant header")
        tenant, role = _membership(app, user, slug)
        if not allowed(role, permission, user.is_super_admin):
            raise HTTPException(403, f"your role cannot do this (needs {permission.value})")
        files = services.tenant_store(app.store, app.settings.master_key, tenant)
        with scoped_session(app.factory, tenant.id) as session:
            session.add(tenant)
            token = credentials.credentials if credentials else ""
            yield Ctx(session, user, tenant, role, files, token, app.settings.controlplane)

    return dependency


def super_admin(app: State, user: CurrentUser) -> Iterator[tuple[Session, User]]:
    if not user.is_super_admin:
        raise HTTPException(403, "only platform super admins can do this")
    with scoped_session(app.factory, ALL_TENANTS) as session:
        yield session, user


def find_or_create_user(session: Session, body: MemberIn) -> User:
    """The account for ``body.email``, created with ``body.password`` if it does not exist."""
    user = session.scalars(select(User).where(User.email == body.email.strip().lower())).first()
    if user is not None:
        return user
    if not body.password:
        raise services.ServiceError("this person has no account yet; set a password")
    return services.create_user(session, body.email, body.password, body.name)
