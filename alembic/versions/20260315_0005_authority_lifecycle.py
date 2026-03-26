"""authority lifecycle fields for review decisions"""
from alembic import op
import sqlalchemy as sa
revision = '20260315_0005'
down_revision = '20260315_0004'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('review_decisions', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('supersedes_review_decision_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('terminates_future_authority', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('terminated_by_review_decision_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_review_decisions_supersedes_review_decision_id', 'review_decisions', ['supersedes_review_decision_id'], ['review_decision_id'])
        batch_op.create_foreign_key('fk_review_decisions_terminated_by_review_decision_id', 'review_decisions', ['terminated_by_review_decision_id'], ['review_decision_id'])
        batch_op.create_index('ix_review_decisions_supersedes_review_decision_id', ['supersedes_review_decision_id'])
        batch_op.create_index('ix_review_decisions_terminated_by_review_decision_id', ['terminated_by_review_decision_id'])
    with op.batch_alter_table('review_group_decisions', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('terminates_future_authority', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('terminated_by_review_group_decision_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_review_group_decisions_terminated_by_review_group_decision_id', 'review_group_decisions', ['terminated_by_review_group_decision_id'], ['review_group_decision_id'])
        batch_op.create_index('ix_review_group_decisions_terminated_by_review_group_decision_id', ['terminated_by_review_group_decision_id'])

def downgrade():
    with op.batch_alter_table('review_group_decisions', recreate='auto') as batch_op:
        batch_op.drop_constraint('fk_review_group_decisions_terminated_by_review_group_decision_id', type_='foreignkey')
        batch_op.drop_index('ix_review_group_decisions_terminated_by_review_group_decision_id')
        batch_op.drop_column('terminated_by_review_group_decision_id')
        batch_op.drop_column('terminates_future_authority')
    with op.batch_alter_table('review_decisions', recreate='auto') as batch_op:
        batch_op.drop_constraint('fk_review_decisions_terminated_by_review_decision_id', type_='foreignkey')
        batch_op.drop_constraint('fk_review_decisions_supersedes_review_decision_id', type_='foreignkey')
        batch_op.drop_index('ix_review_decisions_terminated_by_review_decision_id')
        batch_op.drop_index('ix_review_decisions_supersedes_review_decision_id')
        batch_op.drop_column('terminated_by_review_decision_id')
        batch_op.drop_column('terminates_future_authority')
        batch_op.drop_column('supersedes_review_decision_id')
