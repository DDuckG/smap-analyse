"""scoped review semantics payloads and fingerprints"""
from alembic import op
import sqlalchemy as sa
revision = '20260314_0003'
down_revision = '20260314_0002'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('review_groups', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('semantic_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('scope_key_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('review_context_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('semantic_signature_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('scope_key_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('grouping_policy_payload', sa.JSON(), nullable=True))
        batch_op.create_index('ix_review_groups_semantic_fingerprint', ['semantic_fingerprint'])
        batch_op.create_index('ix_review_groups_scope_key_fingerprint', ['scope_key_fingerprint'])
    with op.batch_alter_table('review_items', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('semantic_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('scope_key_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('review_context_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('semantic_signature_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('scope_key_payload', sa.JSON(), nullable=True))
        batch_op.create_index('ix_review_items_semantic_fingerprint', ['semantic_fingerprint'])
        batch_op.create_index('ix_review_items_scope_key_fingerprint', ['scope_key_fingerprint'])
    with op.batch_alter_table('review_decisions', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('review_context_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('semantic_signature_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('applicability_policy_payload', sa.JSON(), nullable=True))
    with op.batch_alter_table('review_group_decisions', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('review_context_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('semantic_signature_payload', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('applicability_policy_payload', sa.JSON(), nullable=True))

def downgrade():
    with op.batch_alter_table('review_group_decisions', recreate='auto') as batch_op:
        batch_op.drop_column('applicability_policy_payload')
        batch_op.drop_column('semantic_signature_payload')
        batch_op.drop_column('review_context_payload')
    with op.batch_alter_table('review_decisions', recreate='auto') as batch_op:
        batch_op.drop_column('applicability_policy_payload')
        batch_op.drop_column('semantic_signature_payload')
        batch_op.drop_column('review_context_payload')
    with op.batch_alter_table('review_items', recreate='auto') as batch_op:
        batch_op.drop_index('ix_review_items_scope_key_fingerprint')
        batch_op.drop_index('ix_review_items_semantic_fingerprint')
        batch_op.drop_column('scope_key_payload')
        batch_op.drop_column('semantic_signature_payload')
        batch_op.drop_column('review_context_payload')
        batch_op.drop_column('scope_key_fingerprint')
        batch_op.drop_column('semantic_fingerprint')
    with op.batch_alter_table('review_groups', recreate='auto') as batch_op:
        batch_op.drop_index('ix_review_groups_scope_key_fingerprint')
        batch_op.drop_index('ix_review_groups_semantic_fingerprint')
        batch_op.drop_column('grouping_policy_payload')
        batch_op.drop_column('scope_key_payload')
        batch_op.drop_column('semantic_signature_payload')
        batch_op.drop_column('review_context_payload')
        batch_op.drop_column('scope_key_fingerprint')
        batch_op.drop_column('semantic_fingerprint')
