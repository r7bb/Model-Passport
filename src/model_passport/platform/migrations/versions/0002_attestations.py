"""Signed attestations: a public key per tenant, and an attestation per model version.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("tenants") as batch:
        batch.add_column(sa.Column("public_key", sa.Text(), nullable=True))
    with op.batch_alter_table("model_versions") as batch:
        batch.add_column(sa.Column("attestation", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("attestation_sha256", sa.String(length=64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("model_versions") as batch:
        batch.drop_column("attestation_sha256")
        batch.drop_column("attestation")
    with op.batch_alter_table("tenants") as batch:
        batch.drop_column("public_key")
