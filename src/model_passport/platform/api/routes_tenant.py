"""Organization routes, one group per dashboard module.

M1 dashboard, M2 access and roles, M3 analytics, M4 audit logs, M5 model analysis, M6 version
control, M7 pipelines and workers, M9 data provenance and the diligence report. (M8 developer
endpoints arrive with the release flow.)
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select

from model_passport.platform import auditlog, jobs, lifecycle, reports, services
from model_passport.platform.api.deps import Ctx, find_or_create_user, require
from model_passport.platform.api.schemas import (
    ChainStatus,
    DatasetOut,
    DecisionIn,
    EventOut,
    JobOut,
    KillIn,
    MemberIn,
    MemberOut,
    MemberPatch,
    ModelDetail,
    ModelIn,
    ModelOut,
    TestReportIn,
    TestReportOut,
    TransitionIn,
    VersionIn,
    VersionOut,
)
from model_passport.platform.controlplane import ControlPlane, ControlPlaneError
from model_passport.platform.models import (
    Approval,
    AuditEvent,
    Dataset,
    Job,
    Membership,
    Model,
    ModelVersion,
    State,
    TestReport,
    User,
)
from model_passport.platform.rbac import Permission

router = APIRouter()
P = Permission


def ctx(permission: P) -> Any:
    return Depends(require(permission))


def _version(c: Ctx, version_id: str) -> ModelVersion:
    try:
        return services.get(c.session, ModelVersion, c.tenant.id, version_id)
    except services.NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def _model(c: Ctx, model_id: str) -> Model:
    try:
        return services.get(c.session, Model, c.tenant.id, model_id)
    except services.NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def _dataset(c: Ctx, dataset_id: str) -> Dataset:
    try:
        return services.get(c.session, Dataset, c.tenant.id, dataset_id)
    except services.NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


CONTROLPLANE_HTTP = {
    "PERMISSION_DENIED": 403,
    "FAILED_PRECONDITION": 409,
    "NOT_FOUND": 404,
    "UNAUTHENTICATED": 401,
    "INVALID_ARGUMENT": 400,
}
DEPLOYS = {State.CANARY: "dev", State.RELEASED: "consumer"}


def _controlplane(c: Ctx) -> ControlPlane:
    plane = c.controlplane
    if plane is None:
        raise HTTPException(501, "no control plane is configured (set MP_CONTROLPLANE_ADDR)")
    return plane


def _deploy_first(c: Ctx, version: ModelVersion, to: State) -> None:
    """Deploy through the control plane before the state change is written.

    The control plane locks the version row; calling it after this request has written the
    row would deadlock, so the deployment comes first and the state change second.
    """
    plane, environment = c.controlplane, DEPLOYS.get(to)
    if plane is None or environment is None:
        return
    try:
        plane.deploy(version.id, environment)
    except ControlPlaneError as exc:
        raise HTTPException(CONTROLPLANE_HTTP.get(exc.code, 502), str(exc)) from exc


def _move(c: Ctx, version: ModelVersion, to: State, reason: str) -> None:
    try:
        lifecycle.move(c.session, version, to, c.who, reason)
    except lifecycle.TransitionError as exc:
        raise HTTPException(409, str(exc)) from exc


# --- M1 Dashboard ------------------------------------------------------------------------------


@router.get("/dashboard", tags=["M1 dashboard"])
def dashboard(c: Annotated[Ctx, ctx(P.MODELS_READ)]) -> dict[str, Any]:
    versions = list(
        c.session.scalars(select(ModelVersion).where(ModelVersion.tenant_id == c.tenant.id))
    )
    open_jobs = c.session.scalar(
        select(func.count())
        .select_from(Job)
        .where(Job.tenant_id == c.tenant.id, Job.status.in_(["queued", "running"]))
    )
    recent = c.session.scalars(
        select(AuditEvent)
        .where(AuditEvent.tenant_id == c.tenant.id)
        .order_by(AuditEvent.seq.desc())
        .limit(10)
    ).all()
    return {
        "organization": {"slug": c.tenant.slug, "name": c.tenant.name, "plan": c.tenant.plan.value},
        "role": c.role.value if c.role else "super_admin",
        "models": c.session.scalar(
            select(func.count()).select_from(Model).where(Model.tenant_id == c.tenant.id)
        ),
        "versions_by_state": dict(Counter(v.state.value for v in versions)),
        "open_findings": {
            "critical": sum(v.critical for v in versions if v.state is State.FINDINGS),
            "high": sum(v.high for v in versions if v.state is State.FINDINGS),
        },
        "blocked": [v.id for v in versions if v.state is State.KILLED],
        "jobs_in_progress": open_jobs,
        "recent_activity": [
            {"seq": e.seq, "actor": e.actor, "action": e.action, "at": e.created_at.isoformat()}
            for e in recent
        ],
    }


# --- M2 Access and roles -----------------------------------------------------------------------


@router.get("/members", response_model=list[MemberOut], tags=["M2 access & roles"])
def members(c: Annotated[Ctx, ctx(P.MEMBERS_MANAGE)]) -> list[MemberOut]:
    rows = c.session.execute(
        select(User, Membership.role)
        .join(Membership, Membership.user_id == User.id)
        .where(Membership.tenant_id == c.tenant.id)
        .order_by(User.email)
    ).all()
    return [MemberOut(user_id=u.id, email=u.email, name=u.name, role=r) for u, r in rows]


@router.post("/members", response_model=MemberOut, status_code=201, tags=["M2 access & roles"])
def add_member(body: MemberIn, c: Annotated[Ctx, ctx(P.MEMBERS_MANAGE)]) -> MemberOut:
    try:
        user = find_or_create_user(c.session, body)
        services.add_member(c.session, c.tenant, user, body.role, c.actor)
    except services.ServiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    return MemberOut(user_id=user.id, email=user.email, name=user.name, role=body.role)


def _membership(c: Ctx, user_id: str) -> Membership:
    member = c.session.scalars(
        select(Membership).where(Membership.tenant_id == c.tenant.id, Membership.user_id == user_id)
    ).first()
    if member is None:
        raise HTTPException(404, "not a member of this organization")
    return member


@router.patch("/members/{user_id}", response_model=MemberOut, tags=["M2 access & roles"])
def change_role(
    user_id: str, body: MemberPatch, c: Annotated[Ctx, ctx(P.MEMBERS_MANAGE)]
) -> MemberOut:
    member = _membership(c, user_id)
    services.add_member(c.session, c.tenant, member.user, body.role, c.actor)
    return MemberOut(
        user_id=user_id, email=member.user.email, name=member.user.name, role=body.role
    )


@router.delete("/members/{user_id}", status_code=204, tags=["M2 access & roles"])
def remove_member(user_id: str, c: Annotated[Ctx, ctx(P.MEMBERS_MANAGE)]) -> None:
    if user_id == c.user.id:
        raise HTTPException(400, "you cannot remove yourself")
    services.remove_member(c.session, _membership(c, user_id), c.actor)


# --- M3 Analytics ------------------------------------------------------------------------------


@router.get("/analytics", tags=["M3 analytics & reports"])
def analytics(c: Annotated[Ctx, ctx(P.ANALYTICS_READ)]) -> dict[str, Any]:
    versions = list(
        c.session.scalars(
            select(ModelVersion)
            .where(ModelVersion.tenant_id == c.tenant.id)
            .order_by(ModelVersion.created_at)
        )
    )
    by_type: Counter[str] = Counter()
    for version in versions:
        for finding in (version.audit or {}).get("findings", []):
            if finding.get("severity") in ("high", "critical"):
                by_type[finding["entity_type"]] += 1
    return {
        "risk_over_time": [
            {
                "model": v.model.name,
                "version": v.version,
                "auc": v.auc,
                "critical": v.critical,
                "high": v.high,
                "verdict": v.verdict,
                "at": v.created_at.isoformat(),
            }
            for v in versions
            if v.audit
        ],
        "high_risk_findings_by_type": dict(by_type.most_common()),
        "remediation_rounds": sum(1 for v in versions if v.parent_id),
    }


# --- M4 Audit logs -----------------------------------------------------------------------------


@router.get("/audit-log", response_model=list[EventOut], tags=["M4 audit logs"])
def audit_log(
    c: Annotated[Ctx, ctx(P.AUDIT_LOG_READ)], after: int = 0, limit: int = 100
) -> list[AuditEvent]:
    return list(
        c.session.scalars(
            select(AuditEvent)
            .where(AuditEvent.tenant_id == c.tenant.id, AuditEvent.seq > after)
            .order_by(AuditEvent.seq)
            .limit(min(limit, 1000))
        )
    )


@router.get("/audit-log/verify", response_model=ChainStatus, tags=["M4 audit logs"])
def verify_log(c: Annotated[Ctx, ctx(P.AUDIT_LOG_READ)]) -> ChainStatus:
    events = auditlog.chain(c.session, c.tenant.id)
    problems = auditlog.verify_chain(events)
    return ChainStatus(events=len(events), intact=not problems, problems=problems)


# --- M9 Data provenance ------------------------------------------------------------------------


@router.post("/datasets", response_model=DatasetOut, status_code=201, tags=["M9 data provenance"])
def add_dataset(
    c: Annotated[Ctx, ctx(P.MODELS_WRITE)],
    file: Annotated[UploadFile, File(description="Corpus: JSONL with text and entities")],
    name: Annotated[str, Form()],
    source: Annotated[str, Form()] = "",
    license: Annotated[str, Form()] = "unknown",
    consent: Annotated[str, Form()] = "unknown",
) -> Dataset:
    content = file.file.read()
    try:
        return services.add_dataset(
            c.session,
            c.files,
            c.tenant,
            name,
            content,
            services.Provenance(source, license, consent),
            c.actor,
        )
    except services.ServiceError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets", response_model=list[DatasetOut], tags=["M9 data provenance"])
def datasets(c: Annotated[Ctx, ctx(P.PROVENANCE_READ)]) -> list[Dataset]:
    return list(
        c.session.scalars(
            select(Dataset).where(Dataset.tenant_id == c.tenant.id).order_by(Dataset.created_at)
        )
    )


@router.get("/models/{model_id}/diligence", tags=["M9 data provenance"], response_model=None)
def diligence(
    model_id: str, c: Annotated[Ctx, ctx(P.REPORTS_READ)], format: str = "json"
) -> dict[str, Any] | HTMLResponse:
    """The diligence report for investors and acquirers (JSON, or ``?format=html``)."""
    report = reports.diligence(c.session, c.tenant, _model(c, model_id))
    auditlog.record(c.session, c.tenant.id, c.actor, "report.viewed", "model", model_id)
    return HTMLResponse(reports.render_html(report)) if format == "html" else report


# --- M5 Model analysis and M6 version control --------------------------------------------------


@router.post("/models", response_model=ModelOut, status_code=201, tags=["M5 model analysis"])
def register_model(body: ModelIn, c: Annotated[Ctx, ctx(P.MODELS_WRITE)]) -> Model:
    try:
        return services.register_model(
            c.session, c.tenant, body.name, body.access, body.base, body.description, c.actor
        )
    except services.ServiceError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/models", response_model=list[ModelOut], tags=["M5 model analysis"])
def models(c: Annotated[Ctx, ctx(P.MODELS_READ)]) -> list[Model]:
    return list(
        c.session.scalars(select(Model).where(Model.tenant_id == c.tenant.id).order_by(Model.name))
    )


@router.get("/models/{model_id}", response_model=ModelDetail, tags=["M6 version control"])
def model_detail(model_id: str, c: Annotated[Ctx, ctx(P.MODELS_READ)]) -> Model:
    return _model(c, model_id)


@router.post(
    "/models/{model_id}/versions",
    response_model=VersionOut,
    status_code=201,
    tags=["M6 version control"],
)
def register_version(
    model_id: str, body: VersionIn, c: Annotated[Ctx, ctx(P.MODELS_WRITE)]
) -> ModelVersion:
    model = _model(c, model_id)
    training = _dataset(c, body.dataset_id) if body.dataset_id else None
    try:
        version = services.register_version(
            c.session,
            model,
            body.version,
            training,
            _dataset(c, body.reference_dataset_id),
            c.actor,
        )
    except services.ServiceError as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.train:
        if not c.can(P.AUDITS_RUN):
            raise HTTPException(403, "your role cannot start training and audits")
        jobs.enqueue(
            c.session, c.tenant.id, "train", {"version": version.id, "then_audit": True}, c.actor
        )
    return version


@router.get("/versions/{version_id}", response_model=VersionOut, tags=["M6 version control"])
def version_detail(version_id: str, c: Annotated[Ctx, ctx(P.MODELS_READ)]) -> ModelVersion:
    return _version(c, version_id)


@router.get("/versions/{version_id}/findings", tags=["M5 model analysis"])
def findings(version_id: str, c: Annotated[Ctx, ctx(P.FINDINGS_READ)]) -> dict[str, Any]:
    """The entity audit: attack strength and per-entity findings (masked values only)."""
    version = _version(c, version_id)
    if not version.audit:
        raise HTTPException(404, "this version has not been audited yet")
    return {"state": version.state.value, "verdict": version.verdict, "audit": version.audit}


@router.get("/versions/{version_id}/attestation", tags=["M6 version control"])
def attestation(version_id: str, c: Annotated[Ctx, ctx(P.PROVENANCE_READ)]) -> dict[str, Any]:
    version = _version(c, version_id)
    if not version.attestation:
        raise HTTPException(404, "no attestation yet; it is issued when the audit completes")
    return {"attestation": version.attestation, "public_key": c.tenant.public_key}


@router.post(
    "/versions/{version_id}/audit",
    response_model=JobOut,
    status_code=202,
    tags=["M5 model analysis"],
)
def start_audit(version_id: str, c: Annotated[Ctx, ctx(P.AUDITS_RUN)]) -> Job:
    version = _version(c, version_id)
    _move(c, version, State.AUDITING, "audit requested")
    return jobs.enqueue(c.session, c.tenant.id, "audit", {"version": version.id}, c.actor)


@router.post(
    "/versions/{version_id}/remediate",
    response_model=JobOut,
    status_code=202,
    tags=["M7 pipelines & workers"],
)
def start_remediation(version_id: str, c: Annotated[Ctx, ctx(P.REMEDIATION_RUN)]) -> Job:
    version = _version(c, version_id)
    _move(c, version, State.REMEDIATING, "remediation requested")
    return jobs.enqueue(c.session, c.tenant.id, "remediate", {"version": version.id}, c.actor)


@router.post(
    "/versions/{version_id}/transition", response_model=VersionOut, tags=["M6 version control"]
)
def transition(
    version_id: str, body: TransitionIn, c: Annotated[Ctx, ctx(P.MODELS_READ)]
) -> ModelVersion:
    """Any lifecycle step the caller's role allows (the lifecycle checks the permission).

    With a control plane, ``canary`` deploys to developer endpoints and ``released`` to consumers.
    """
    version = _version(c, version_id)
    try:
        lifecycle.check(version, body.to, c.who)  # same answer with or without a control plane
    except lifecycle.TransitionError as exc:
        raise HTTPException(409, str(exc)) from exc
    _deploy_first(c, version, body.to)
    _move(c, version, body.to, body.reason)
    return version


def _check_approvable(c: Ctx, version: ModelVersion) -> None:
    if version.verdict == "fail" or version.critical or version.high:
        raise HTTPException(409, "the latest audit still has blocking findings; remediate first")
    leaks = c.session.scalar(
        select(func.coalesce(func.sum(TestReport.leaks_found), 0)).where(
            TestReport.version_id == version.id
        )
    )
    if leaks:
        raise HTTPException(409, "canary testers reported leaks; remediate before approving")


@router.post(
    "/versions/{version_id}/approvals", response_model=VersionOut, tags=["M2 access & roles"]
)
def decide(
    version_id: str, body: DecisionIn, c: Annotated[Ctx, ctx(P.APPROVALS_DECIDE)]
) -> ModelVersion:
    """Sign off (or reject) a verified version.

    Approval needs a passing, clean audit and no leaks reported by canary testers.
    """
    version = _version(c, version_id)
    if body.decision == "approve":
        _check_approvable(c, version)
    c.session.add(
        Approval(
            tenant_id=c.tenant.id,
            version_id=version.id,
            user_id=c.user.id,
            role=c.role or "org_admin",
            decision=body.decision,
            comment=body.comment,
        )
    )
    _move(
        c, version, State.APPROVED if body.decision == "approve" else State.REJECTED, body.comment
    )
    return version


@router.post("/versions/{version_id}/kill", response_model=VersionOut, tags=["M1 dashboard"])
def kill(version_id: str, body: KillIn, c: Annotated[Ctx, ctx(P.KILL_SWITCH)]) -> ModelVersion:
    """The kill switch: block this version everywhere, immediately.

    With a control plane, every live deployment of the version stops at the same moment.
    """
    version = _version(c, version_id)
    plane = c.controlplane
    if plane is None:
        _move(c, version, State.KILLED, body.reason)
        return version
    try:
        plane.kill(version.id, body.reason)
    except ControlPlaneError as exc:
        raise HTTPException(CONTROLPLANE_HTTP.get(exc.code, 502), str(exc)) from exc
    c.session.refresh(version)
    return version


@router.post("/models/{model_id}/rollback", tags=["M8 dev endpoints & testing"])
def rollback(model_id: str, c: Annotated[Ctx, ctx(P.RELEASES_MANAGE)]) -> dict[str, Any]:
    """Retire the live consumer version and bring back the previous release."""
    try:
        return _controlplane(c).rollback(_model(c, model_id).id)
    except ControlPlaneError as exc:
        raise HTTPException(CONTROLPLANE_HTTP.get(exc.code, 502), str(exc)) from exc


@router.get("/deployments", tags=["M8 dev endpoints & testing"])
def deployments(c: Annotated[Ctx, ctx(P.MODELS_READ)], model_id: str = "") -> list[dict[str, Any]]:
    """What serves where: developer endpoints and consumer releases."""
    try:
        return _controlplane(c).deployments(model_id)
    except ControlPlaneError as exc:
        raise HTTPException(CONTROLPLANE_HTTP.get(exc.code, 502), str(exc)) from exc


@router.post(
    "/models/{model_id}/reports",
    response_model=JobOut,
    status_code=202,
    tags=["M3 analytics & reports"],
)
def generate_report(model_id: str, c: Annotated[Ctx, ctx(P.REPORTS_READ)]) -> Job:
    """Generate and store the diligence report (JSON and HTML), encrypted."""
    return jobs.enqueue(
        c.session, c.tenant.id, "report", {"model": _model(c, model_id).id}, c.actor
    )


# --- M8 Developer endpoints and testing ---------------------------------------------------------


@router.post(
    "/versions/{version_id}/test-reports",
    response_model=TestReportOut,
    status_code=201,
    tags=["M8 dev endpoints & testing"],
)
def submit_test_report(
    version_id: str, body: TestReportIn, c: Annotated[Ctx, ctx(P.TEST_REPORTS)]
) -> TestReport:
    """A canary tester's extraction attempts against a version on developer endpoints."""
    version = _version(c, version_id)
    if version.state not in (State.CANARY, State.VERIFYING):
        raise HTTPException(409, "test reports are for versions on developer endpoints")
    report = TestReport(
        tenant_id=c.tenant.id,
        version_id=version.id,
        user_id=c.user.id,
        prompt_count=body.prompt_count,
        leaks_found=body.leaks_found,
        summary=body.summary,
    )
    c.session.add(report)
    c.session.flush()
    auditlog.record(
        c.session,
        c.tenant.id,
        c.actor,
        "testreport.submitted",
        "model_version",
        version.id,
        {"prompts": body.prompt_count, "leaks": body.leaks_found},
    )
    return report


@router.get(
    "/versions/{version_id}/test-reports",
    response_model=list[TestReportOut],
    tags=["M8 dev endpoints & testing"],
)
def test_reports(version_id: str, c: Annotated[Ctx, ctx(P.MODELS_READ)]) -> list[TestReport]:
    version = _version(c, version_id)
    return list(
        c.session.scalars(
            select(TestReport)
            .where(TestReport.version_id == version.id)
            .order_by(TestReport.created_at)
        )
    )


# --- M7 Pipelines and workers ------------------------------------------------------------------


@router.get("/jobs", response_model=list[JobOut], tags=["M7 pipelines & workers"])
def list_jobs(c: Annotated[Ctx, ctx(P.JOBS_READ)], limit: int = 50) -> list[Job]:
    return list(
        c.session.scalars(
            select(Job)
            .where(Job.tenant_id == c.tenant.id)
            .order_by(Job.created_at.desc())
            .limit(min(limit, 500))
        )
    )


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["M7 pipelines & workers"])
def job_detail(job_id: str, c: Annotated[Ctx, ctx(P.JOBS_READ)]) -> Job:
    job = c.session.get(Job, job_id)
    if job is None or job.tenant_id != c.tenant.id:
        raise HTTPException(404, "job not found")
    return job
