"""Platform core: security, encrypted storage, audit chain, lifecycle, jobs, and services."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from model_passport.llm import synthetic
from model_passport.llm.entities import write_corpus
from model_passport.platform import auditlog, jobs, lifecycle, services
from model_passport.platform.db import make_engine, migrate, scoped_session, session_factory
from model_passport.platform.models import Access, Job, JobStatus, Model, ModelVersion, Role, State
from model_passport.platform.rbac import Permission, allowed
from model_passport.platform.security import (
    DecryptionError,
    TokenError,
    decrypt,
    encrypt,
    hash_password,
    issue_token,
    read_token,
    unwrap_key,
    verify_password,
)
from model_passport.platform.storage import LocalStore, StorageError, TenantStore

MASTER = os.urandom(32)
ADMIN = auditlog.Actor("admin-id", "admin@example.com")
SECRET = "x" * 40


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    engine = make_engine(f"sqlite:///{tmp_path}/mp.db")
    migrate(engine)
    with scoped_session(session_factory(engine), "*") as s:
        yield s


# --- Security ----------------------------------------------------------------------------------


def test_passwords_and_tokens() -> None:
    stored = hash_password("correct horse battery")
    assert stored.startswith("$argon2id$")
    assert verify_password(stored, "correct horse battery")
    assert not verify_password(stored, "wrong")
    token = issue_token("u1", False, SECRET, minutes=5)
    assert read_token(token, SECRET)["sub"] == "u1"
    with pytest.raises(TokenError):
        read_token(token, "y" * 40)
    with pytest.raises(TokenError):
        read_token(issue_token("u1", False, SECRET, minutes=-1), SECRET)


def test_encryption_is_bound_to_key_and_context() -> None:
    key = os.urandom(32)
    blob = encrypt(key, b"secret", b"tenant-a/obj")
    assert decrypt(key, blob, b"tenant-a/obj") == b"secret"
    with pytest.raises(DecryptionError):
        decrypt(key, blob, b"tenant-b/obj")
    with pytest.raises(DecryptionError):
        decrypt(os.urandom(32), blob, b"tenant-a/obj")


def test_tenant_store_encrypts_and_isolates(tmp_path: Path) -> None:
    raw = LocalStore(tmp_path)
    a = TenantStore(raw, "tenant-a", os.urandom(32))
    b = TenantStore(raw, "tenant-b", os.urandom(32))
    a.put("datasets/x.jsonl", b"Mail ann@example.com")
    assert a.get("datasets/x.jsonl") == b"Mail ann@example.com"
    on_disk = (tmp_path / "tenant-a/datasets/x.jsonl").read_bytes()
    assert b"ann@example.com" not in on_disk  # encrypted at rest
    (tmp_path / "tenant-b/datasets").mkdir(parents=True)
    (tmp_path / "tenant-b/datasets/x.jsonl").write_bytes(on_disk)  # copy across tenants
    with pytest.raises(DecryptionError):
        b.get("datasets/x.jsonl")
    with pytest.raises(StorageError):
        a.get("../tenant-b/datasets/x.jsonl")


def test_roles_follow_the_product_flow() -> None:
    assert allowed(Role.COMPLIANCE_AUDITOR, Permission.KILL_SWITCH)
    assert allowed(Role.ML_ENGINEER, Permission.REMEDIATION_RUN)
    assert not allowed(Role.ML_ENGINEER, Permission.APPROVALS_DECIDE)
    assert allowed(Role.EXTERNAL_REVIEWER, Permission.REPORTS_READ)
    assert not allowed(Role.EXTERNAL_REVIEWER, Permission.FINDINGS_READ)
    assert not allowed(Role.END_CONSUMER, Permission.MODELS_READ)
    assert allowed(None, Permission.TENANTS_MANAGE, super_admin=True)
    assert not allowed(Role.ORG_ADMIN, Permission.TENANTS_MANAGE)


# --- Audit log ---------------------------------------------------------------------------------


def test_audit_chain_detects_tampering(session: Session) -> None:
    for i in range(3):
        auditlog.record(session, "t1", ADMIN, "thing.done", "thing", str(i), {"i": i})
    auditlog.record(session, "t2", ADMIN, "other.done")
    events = auditlog.chain(session, "t1")
    assert [e.seq for e in events] == [1, 2, 3]
    assert auditlog.verify_chain(events) == []
    events[1].details = {"i": 99}  # edited in memory, as an attacker with DB access would
    assert auditlog.verify_chain(events) == ["event 2: contents changed since it was recorded"]
    session.expire_all()
    with pytest.raises(Exception, match="append-only"):
        session.execute(text("UPDATE audit_events SET action = 'x'"))


# --- Services, lifecycle, jobs -----------------------------------------------------------------


def _tenant_with_model(
    session: Session, tmp_path: Path
) -> tuple[services.Tenant, Model, TenantStore]:
    tenant = services.create_tenant(session, MASTER, "usps", "USPS", ADMIN)
    files = services.tenant_store(LocalStore(tmp_path / "objects"), MASTER, tenant)
    corpus = tmp_path / "c.jsonl"
    write_corpus(synthetic.corpus(20, seed=0), corpus)
    dataset = services.add_dataset(
        session, files, tenant, "tickets-2025", corpus.read_bytes(),
        services.Provenance("support desk export", "internal", "obtained"), ADMIN,
    )  # fmt: skip
    model = services.register_model(session, tenant, "assistant", Access.OPEN_WEIGHT,
                                    "hf:EleutherAI/pythia-160m", "", ADMIN)  # fmt: skip
    services.register_version(session, model, "1.0.0", dataset, dataset, ADMIN)
    return tenant, model, files


def test_tenants_datasets_and_versions(session: Session, tmp_path: Path) -> None:
    tenant, model, files = _tenant_with_model(session, tmp_path)
    assert unwrap_key(MASTER, tenant.wrapped_key, tenant.id) == files.key
    with pytest.raises(services.ServiceError, match="slug"):
        services.create_tenant(session, MASTER, "Bad Slug!", "x", ADMIN)
    with pytest.raises(services.ServiceError, match="already exists"):
        services.create_tenant(session, MASTER, "usps", "again", ADMIN)
    with pytest.raises(services.ServiceError, match="12 characters"):
        services.create_user(session, "a@b.co", "short")
    user = services.create_user(session, "eng@usps.example", "a long enough password")
    services.add_member(session, tenant, user, Role.ML_ENGINEER, ADMIN)
    services.add_member(session, tenant, user, Role.COMPLIANCE_AUDITOR, ADMIN)  # role change
    assert [m.role for m in user.memberships] == [Role.COMPLIANCE_AUDITOR]
    version = model.versions[0]
    assert services.get(session, Model, tenant.id, model.id) is model
    with pytest.raises(services.NotFoundError):
        services.get(session, Model, "another-tenant", model.id)
    actions = [e.action for e in auditlog.chain(session, tenant.id)]
    assert actions[:3] == ["dataset.added", "model.registered", "version.registered"]
    assert "member.role_changed" in actions
    assert version.state is State.REGISTERED


def test_lifecycle_enforces_steps_and_roles(session: Session, tmp_path: Path) -> None:
    _, model, _ = _tenant_with_model(session, tmp_path)
    version = model.versions[0]
    engineer = lifecycle.Who(ADMIN, Role.ML_ENGINEER)
    auditor = lifecycle.Who(ADMIN, Role.COMPLIANCE_AUDITOR)
    worker = lifecycle.Who(auditlog.SYSTEM, None, system=True)
    with pytest.raises(lifecycle.TransitionError, match="not a lifecycle step"):
        lifecycle.move(session, version, State.RELEASED, engineer)
    lifecycle.move(session, version, State.AUDITING, engineer)
    with pytest.raises(lifecycle.TransitionError, match="set by the platform"):
        lifecycle.move(session, version, State.CLEAN, engineer)
    lifecycle.move(session, version, State.CLEAN, worker, "no confirmed high-risk entities")
    lifecycle.move(session, version, State.CANARY, engineer)
    lifecycle.move(session, version, State.VERIFYING, engineer)
    with pytest.raises(lifecycle.TransitionError, match=r"approvals\.decide"):
        lifecycle.move(session, version, State.APPROVED, engineer)
    lifecycle.move(session, version, State.APPROVED, auditor)
    lifecycle.move(session, version, State.KILLED, auditor, "leak reported")
    assert version.state is State.KILLED
    assert lifecycle.next_states(State.KILLED) == []


def test_jobs_are_claimed_once_and_retried(session: Session, tmp_path: Path) -> None:
    tenant, model, _ = _tenant_with_model(session, tmp_path)
    job = jobs.enqueue(session, tenant.id, "audit", {"version": model.versions[0].id}, ADMIN,
                       max_attempts=2)  # fmt: skip
    with pytest.raises(jobs.JobError):
        jobs.enqueue(session, tenant.id, "mine-bitcoin", {}, ADMIN)
    claimed = jobs.claim(session, "w1")
    assert claimed is job
    assert jobs.claim(session, "w2") is None  # already running
    jobs.fail(session, job, "boom")
    assert job.status is JobStatus.QUEUED  # retried
    again = jobs.claim(session, "w2")
    assert again is job
    jobs.fail(session, job, "boom again")
    assert job.status is JobStatus.FAILED
    assert job.attempts == 2


# --- PostgreSQL row-level security ---------------------------------------------------------------


def test_postgres_row_level_security_isolates_tenants() -> None:
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tempfile.mkdtemp())
    server.psql("CREATE ROLE mp_app LOGIN PASSWORD 'pw'; CREATE DATABASE mp OWNER mp_app;")
    host = server.get_uri().split("host=")[1]
    engine = make_engine(f"postgresql+psycopg://mp_app:pw@/mp?host={host}")
    migrate(engine)
    factory = session_factory(engine)
    with scoped_session(factory, "*") as s:
        a = services.create_tenant(s, MASTER, "usps", "USPS", ADMIN)
        b = services.create_tenant(s, MASTER, "ups", "UPS", ADMIN)
        services.register_model(s, a, "a-model", Access.API, "anthropic:claude-sonnet-5", "", ADMIN)
        services.register_model(s, b, "b-model", Access.API, "openai:gpt-5", "", ADMIN)
        a_id, b_id = a.id, b.id
    with scoped_session(factory, a_id) as s:
        assert [m.name for m in s.query(Model).all()] == ["a-model"]
    with pytest.raises(Exception, match="row-level security"), scoped_session(factory, a_id) as s:
        s.add(Model(tenant_id=b_id, name="sneaky", access=Access.API, base="x"))
    with engine.connect() as connection:  # no tenant set: nothing is visible
        assert connection.execute(text("SELECT count(*) FROM models")).scalar() == 0


# --- Settings, storage backends, and worker failures ---------------------------------------------


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from model_passport.platform.settings import Settings, SettingsError, decode_key, new_key

    monkeypatch.setenv("MP_JWT_SECRET", "short")
    with pytest.raises(SettingsError, match="MP_JWT_SECRET"):
        Settings.from_env()
    monkeypatch.setenv("MP_JWT_SECRET", "j" * 40)
    monkeypatch.delenv("MP_MASTER_KEY", raising=False)
    with pytest.raises(SettingsError, match="MP_MASTER_KEY"):
        Settings.from_env()
    monkeypatch.setenv("MP_MASTER_KEY", new_key())
    monkeypatch.setenv("MP_BASE_DOMAIN", "mp.example.com")
    settings = Settings.from_env()
    assert settings.base_domain == "mp.example.com"
    assert len(settings.master_key) == 32
    with pytest.raises(SettingsError, match="base64"):
        decode_key("not base64!!")
    with pytest.raises(SettingsError, match="32 bytes"):
        decode_key("c2hvcnQ=")


def test_store_urls_and_s3_backend(tmp_path: Path) -> None:
    from model_passport.platform.storage import S3Store, open_store

    assert isinstance(open_store(f"file://{tmp_path}"), LocalStore)
    with pytest.raises(StorageError, match="unsupported"):
        open_store("ftp://host/bucket")

    class FakeS3:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803
            self.objects[f"{Bucket}/{Key}"] = Body

        def get_object(self, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803
            import io

            return {"Body": io.BytesIO(self.objects[f"{Bucket}/{Key}"])}

        def delete_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
            self.objects.pop(f"{Bucket}/{Key}", None)

    store = S3Store("mp-data", endpoint="http://seaweedfs:8333")
    store.client = FakeS3()
    tenant = TenantStore(store, "t1", os.urandom(32))
    tenant.put("reports/a.json", b"{}")
    assert tenant.get("reports/a.json") == b"{}"
    tenant.delete("reports/a.json")
    with pytest.raises(StorageError, match="not found"):
        tenant.get("reports/a.json")


def test_failed_jobs_return_versions_to_an_actionable_state(
    session: Session, tmp_path: Path
) -> None:
    from model_passport.platform.db import make_engine, migrate, session_factory
    from model_passport.platform.worker import WorkerContext, run_once

    engine = make_engine(f"sqlite:///{tmp_path}/w.db")
    migrate(engine)
    factory = session_factory(engine)
    with scoped_session(factory, "*") as s:
        _, model, _ = _tenant_with_model(s, tmp_path)
        version = model.versions[0]
        lifecycle.move(s, version, State.AUDITING, lifecycle.Who(ADMIN, Role.ML_ENGINEER))
        # A tiny base model with no trained checkpoint cannot be audited: the job must fail.
        model.base = "tiny"
        jobs.enqueue(s, version.tenant_id, "audit", {"version": version.id}, ADMIN, max_attempts=1)
        version_id = version.id
    context = WorkerContext(
        factory=factory, store=LocalStore(tmp_path / "objects"), master_key=MASTER
    )
    assert run_once(context) is True
    assert run_once(context) is False  # queue empty
    with scoped_session(factory, "*") as s:
        failed = s.get(Job, s.scalars(select(Job)).first().id)  # type: ignore[union-attr]
        assert failed.status is JobStatus.FAILED
        assert "must be trained" in (failed.error or "")
        assert s.get(ModelVersion, version_id).state is State.REGISTERED  # type: ignore[union-attr]
