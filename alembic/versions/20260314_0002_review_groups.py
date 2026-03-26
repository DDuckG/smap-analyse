"""review groups and typed review signatures"""
from alembic import op
import sqlalchemy as sa
revision = '20260314_0002'
down_revision = '20260314_0001'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('review_groups', sa.Column('review_group_id', sa.Integer(), primary_key=True, autoincrement=True), sa.Column('problem_class', sa.String(length=80), nullable=False), sa.Column('review_signature', sa.String(length=255), nullable=False), sa.Column('normalized_candidate_text', sa.String(length=255), nullable=True), sa.Column('entity_type_hint', sa.String(length=80), nullable=True), sa.Column('ambiguity_signature', sa.String(length=255), nullable=True), sa.Column('candidate_canonical_ids', sa.JSON(), nullable=False), sa.Column('representative_payload', sa.JSON(), nullable=False), sa.Column('occurrence_count', sa.Integer(), nullable=False), sa.Column('active_item_count', sa.Integer(), nullable=False), sa.Column('status', sa.String(length=50), nullable=False), sa.Column('assignee', sa.String(length=120), nullable=True), sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=True), sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True), sa.Column('created_at', sa.DateTime(timezone=True), nullable=False), sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False), sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint('review_signature'))
    op.create_index('ix_review_groups_problem_class', 'review_groups', ['problem_class'])
    op.create_index('ix_review_groups_review_signature', 'review_groups', ['review_signature'])
    op.create_index('ix_review_groups_status', 'review_groups', ['status'])
    op.create_index('ix_review_groups_normalized_candidate_text', 'review_groups', ['normalized_candidate_text'])
    with op.batch_alter_table('review_items', recreate='auto') as batch_op:
        batch_op.add_column(sa.Column('problem_class', sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column('review_signature', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('normalized_candidate_text', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('entity_type_hint', sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column('ambiguity_signature', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('review_group_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('assignee', sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.create_foreign_key('fk_review_items_review_group_id', 'review_groups', ['review_group_id'], ['review_group_id'])
        batch_op.create_index('ix_review_items_problem_class', ['problem_class'])
        batch_op.create_index('ix_review_items_review_signature', ['review_signature'])
        batch_op.create_index('ix_review_items_normalized_candidate_text', ['normalized_candidate_text'])
        batch_op.create_index('ix_review_items_review_group_id', ['review_group_id'])
    op.execute("\n        UPDATE review_items\n        SET\n            problem_class = CASE\n                WHEN item_type = 'entity' THEN 'unresolved_entity_candidate'\n                ELSE 'low_confidence_classification'\n            END,\n            review_signature = 'legacy::' || review_item_id\n        ")
    with op.batch_alter_table('review_items', recreate='auto') as batch_op:
        batch_op.alter_column('problem_class', nullable=False)
        batch_op.alter_column('review_signature', nullable=False)
    op.add_column('review_decisions', sa.Column('resolution_scope', sa.String(length=80), nullable=False, server_default='item_only'))
    op.create_table('review_group_decisions', sa.Column('review_group_decision_id', sa.Integer(), primary_key=True, autoincrement=True), sa.Column('review_group_id', sa.Integer(), sa.ForeignKey('review_groups.review_group_id'), nullable=False), sa.Column('action', sa.String(length=50), nullable=False), sa.Column('reviewer', sa.String(length=100), nullable=False), sa.Column('notes', sa.Text(), nullable=True), sa.Column('remap_target', sa.String(length=200), nullable=True), sa.Column('resolution_scope', sa.String(length=80), nullable=False), sa.Column('created_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('ix_review_group_decisions_review_group_id', 'review_group_decisions', ['review_group_id'])

def downgrade():
    op.drop_index('ix_review_group_decisions_review_group_id', table_name='review_group_decisions')
    op.drop_table('review_group_decisions')
    op.drop_column('review_decisions', 'resolution_scope')
    with op.batch_alter_table('review_items', recreate='auto') as batch_op:
        batch_op.drop_index('ix_review_items_review_group_id')
        batch_op.drop_index('ix_review_items_normalized_candidate_text')
        batch_op.drop_index('ix_review_items_review_signature')
        batch_op.drop_index('ix_review_items_problem_class')
        batch_op.drop_constraint('fk_review_items_review_group_id', type_='foreignkey')
        batch_op.drop_column('resolved_at')
        batch_op.drop_column('assigned_at')
        batch_op.drop_column('assignee')
        batch_op.drop_column('review_group_id')
        batch_op.drop_column('ambiguity_signature')
        batch_op.drop_column('entity_type_hint')
        batch_op.drop_column('normalized_candidate_text')
        batch_op.drop_column('review_signature')
        batch_op.drop_column('problem_class')
    op.drop_index('ix_review_groups_normalized_candidate_text', table_name='review_groups')
    op.drop_index('ix_review_groups_status', table_name='review_groups')
    op.drop_index('ix_review_groups_review_signature', table_name='review_groups')
    op.drop_index('ix_review_groups_problem_class', table_name='review_groups')
    op.drop_table('review_groups')
