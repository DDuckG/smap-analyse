"""applicability truth and decision-chain metadata"""
from alembic import op
import sqlalchemy as sa
revision = '20260315_0004'
down_revision = '20260314_0003'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('review_groups', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('static_scope_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('knowledge_state_fingerprint', sa.String(length=255), nullable=True))
        batch_op.create_index('ix_review_groups_static_scope_fingerprint', ['static_scope_fingerprint'])
        batch_op.create_index('ix_review_groups_knowledge_state_fingerprint', ['knowledge_state_fingerprint'])
    with op.batch_alter_table('review_items', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('static_scope_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('knowledge_state_fingerprint', sa.String(length=255), nullable=True))
        batch_op.create_index('ix_review_items_static_scope_fingerprint', ['static_scope_fingerprint'])
        batch_op.create_index('ix_review_items_knowledge_state_fingerprint', ['knowledge_state_fingerprint'])
    with op.batch_alter_table('review_decisions', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('applicability_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('origin_group_decision_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('effect_applied_from_group_decision_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_review_decisions_origin_group_decision_id', 'review_group_decisions', ['origin_group_decision_id'], ['review_group_decision_id'])
        batch_op.create_foreign_key('fk_review_decisions_effect_applied_from_group_decision_id', 'review_group_decisions', ['effect_applied_from_group_decision_id'], ['review_group_decision_id'])
        batch_op.create_index('ix_review_decisions_applicability_fingerprint', ['applicability_fingerprint'])
        batch_op.create_index('ix_review_decisions_origin_group_decision_id', ['origin_group_decision_id'])
        batch_op.create_index('ix_review_decisions_effect_applied_from_group_decision_id', ['effect_applied_from_group_decision_id'])
    with op.batch_alter_table('review_group_decisions', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('applicability_fingerprint', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('supersedes_review_group_decision_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_review_group_decisions_supersedes_review_group_decision_id', 'review_group_decisions', ['supersedes_review_group_decision_id'], ['review_group_decision_id'])
        batch_op.create_index('ix_review_group_decisions_applicability_fingerprint', ['applicability_fingerprint'])
        batch_op.create_index('ix_review_group_decisions_supersedes_review_group_decision_id', ['supersedes_review_group_decision_id'])

def downgrade():
    with op.batch_alter_table('review_group_decisions', recreate='auto') as batch_op:
        batch_op.drop_constraint('fk_review_group_decisions_supersedes_review_group_decision_id', type_='foreignkey')
        batch_op.drop_index('ix_review_group_decisions_supersedes_review_group_decision_id')
        batch_op.drop_index('ix_review_group_decisions_applicability_fingerprint')
        batch_op.drop_column('supersedes_review_group_decision_id')
        batch_op.drop_column('applicability_fingerprint')
    with op.batch_alter_table('review_decisions', recreate='auto') as batch_op:
        batch_op.drop_constraint('fk_review_decisions_effect_applied_from_group_decision_id', type_='foreignkey')
        batch_op.drop_constraint('fk_review_decisions_origin_group_decision_id', type_='foreignkey')
        batch_op.drop_index('ix_review_decisions_effect_applied_from_group_decision_id')
        batch_op.drop_index('ix_review_decisions_origin_group_decision_id')
        batch_op.drop_index('ix_review_decisions_applicability_fingerprint')
        batch_op.drop_column('effect_applied_from_group_decision_id')
        batch_op.drop_column('origin_group_decision_id')
        batch_op.drop_column('applicability_fingerprint')
    with op.batch_alter_table('review_items', recreate='auto') as batch_op:
        batch_op.drop_index('ix_review_items_knowledge_state_fingerprint')
        batch_op.drop_index('ix_review_items_static_scope_fingerprint')
        batch_op.drop_column('knowledge_state_fingerprint')
        batch_op.drop_column('static_scope_fingerprint')
    with op.batch_alter_table('review_groups', recreate='auto') as batch_op:
        batch_op.drop_index('ix_review_groups_knowledge_state_fingerprint')
        batch_op.drop_index('ix_review_groups_static_scope_fingerprint')
        batch_op.drop_column('knowledge_state_fingerprint')
        batch_op.drop_column('static_scope_fingerprint')
