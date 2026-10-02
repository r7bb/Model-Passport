"""MP Guard: API keys, settings, and the request log, each limited to its organization.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

TABLES = ("guard_keys", "guard_settings", "guard_events")
VISIBLE = (
    "current_setting('app.tenant_id', true) = '*' "
    "OR tenant_id = current_setting('app.tenant_id', true)"
)


def _tenant() -> sa.Column[str]:
    return sa.Column(
        "tenant_id",
        sa.String(length=36),
        sa.ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "guard_keys",
        sa.Column("id", sa.String(length=36), primary_key=True),
        _tenant(),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("hint", sa.String(length=32), nullable=False),
        sa.Column("secret_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "guard_settings",
        sa.Column(
            "tenant_id",
            sa.String(length=36),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("upstream_url", sa.String(length=500), nullable=False),
        sa.Column("upstream_key", sa.LargeBinary(), nullable=True),
        sa.Column("default_model", sa.String(length=200), nullable=False),
        sa.Column("policy", sa.JSON(), nullable=False),
        sa.Column("updated_by", sa.String(length=36), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "guard_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        _tenant(),
        sa.Column("key_id", sa.String(length=36), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("upstream_status", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("report", sa.JSON(), nullable=False),
        sa.Column("detail", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_guard_events_created_at", "guard_events", ["created_at"])
    if op.get_bind().dialect.name == "postgresql":
        for table in TABLES:
            op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
            op.execute(
                f"CREATE POLICY tenant_isolation ON {table} USING ({VISIBLE}) "
                f"WITH CHECK ({VISIBLE})"
            )


def downgrade() -> None:
    op.drop_index("ix_guard_events_created_at", table_name="guard_events")
    for table in reversed(TABLES):
        op.drop_table(table)
