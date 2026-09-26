"""Sign-in, the current user, and the super-admin console (organizations and subscriptions)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from model_passport.platform import auditlog, services
from model_passport.platform.api.deps import CurrentUser, State, find_or_create_user, super_admin
from model_passport.platform.api.schemas import (
    ChainStatus,
    EventOut,
    Login,
    Me,
    MemberIn,
    MemberOut,
    MembershipOut,
    TenantIn,
    TenantOut,
    TenantPatch,
    Token,
)
from model_passport.platform.db import ALL_TENANTS, scoped_session
from model_passport.platform.models import AuditEvent, Membership, Role, Tenant, User
from model_passport.platform.security import issue_token, verify_password

router = APIRouter()
Admin = Annotated[tuple[Session, User], Depends(super_admin)]


@router.post("/auth/login", response_model=Token, tags=["auth"])
def login(body: Login, app: State) -> Token:
    with scoped_session(app.factory, ALL_TENANTS) as session:
        user = session.scalars(select(User).where(User.email == body.email.strip().lower())).first()
        if user is None or user.disabled or not verify_password(user.password_hash, body.password):
            raise HTTPException(401, "wrong email or password")
        minutes = app.settings.token_minutes
        token = issue_token(user.id, user.is_super_admin, app.settings.jwt_secret, minutes)
    return Token(access_token=token, expires_in_minutes=minutes)


@router.get("/me", response_model=Me, tags=["auth"])
def me(app: State, user: CurrentUser) -> Me:
    with scoped_session(app.factory, ALL_TENANTS) as session:
        rows = session.execute(
            select(Tenant.slug, Membership.role)
            .join(Membership, Membership.tenant_id == Tenant.id)
            .where(Membership.user_id == user.id)
        ).all()
    return Me(
        id=user.id,
        email=user.email,
        name=user.name,
        super_admin=user.is_super_admin,
        memberships=[MembershipOut(tenant=slug, role=role) for slug, role in rows],
    )


@router.get("/platform/tenants", response_model=list[TenantOut], tags=["super admin"])
def list_tenants(admin: Admin) -> list[Tenant]:
    return list(admin[0].scalars(select(Tenant).order_by(Tenant.created_at)))


@router.post("/platform/tenants", response_model=TenantOut, status_code=201, tags=["super admin"])
def create_tenant(body: TenantIn, admin: Admin, app: State) -> Tenant:
    session, user = admin
    try:
        return services.create_tenant(
            session,
            app.settings.master_key,
            body.slug,
            body.name,
            auditlog.Actor(user.id, user.email),
            body.plan,
        )
    except services.ServiceError as exc:
        raise HTTPException(400, str(exc)) from exc


def _tenant(session: Session, slug: str) -> Tenant:
    tenant = session.scalars(select(Tenant).where(Tenant.slug == slug)).first()
    if tenant is None:
        raise HTTPException(404, f"organization {slug!r} not found")
    return tenant


@router.patch("/platform/tenants/{slug}", response_model=TenantOut, tags=["super admin"])
def update_tenant(slug: str, body: TenantPatch, admin: Admin) -> Tenant:
    session, user = admin
    return services.update_tenant(
        session, _tenant(session, slug), auditlog.Actor(user.id, user.email), body.plan, body.status
    )


@router.post(
    "/platform/tenants/{slug}/admins",
    response_model=MemberOut,
    status_code=201,
    tags=["super admin"],
)
def add_org_admin(slug: str, body: MemberIn, admin: Admin) -> MemberOut:
    """Create (or promote) an organization's first admin."""
    session, user = admin
    tenant = _tenant(session, slug)
    try:
        target = find_or_create_user(session, body)
        services.add_member(
            session, tenant, target, Role.ORG_ADMIN, auditlog.Actor(user.id, user.email)
        )
    except services.ServiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    return MemberOut(user_id=target.id, email=target.email, name=target.name, role=Role.ORG_ADMIN)


@router.get("/platform/audit-log", response_model=list[EventOut], tags=["super admin"])
def platform_log(admin: Admin) -> list[AuditEvent]:
    return auditlog.chain(admin[0], auditlog.PLATFORM)


@router.get("/platform/audit-log/verify", response_model=ChainStatus, tags=["super admin"])
def verify_platform_log(admin: Admin) -> ChainStatus:
    events = auditlog.chain(admin[0], auditlog.PLATFORM)
    problems = auditlog.verify_chain(events)
    return ChainStatus(events=len(events), intact=not problems, problems=problems)
