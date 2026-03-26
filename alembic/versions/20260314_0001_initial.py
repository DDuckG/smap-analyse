"""initial review and ontology tables"""
from alembic import op
import sqlalchemy as sa
revision = '20260314_0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('ontology_registry_versions', sa.Column('version_id', sa.String(length=100), primary_key=True), sa.Column('name', sa.String(length=100), nullable=False), sa.Column('description', sa.Text(), nullable=False), sa.Column('applied_at', sa.DateTime(timezone=True), nullable=False))
    op.create_table('review_items', sa.Column('review_item_id', sa.Integer(), primary_key=True, autoincrement=True), sa.Column('item_type', sa.String(length=50), nullable=False), sa.Column('mention_id', sa.String(length=200), nullable=False), sa.Column('source_uap_id', sa.String(length=200), nullable=False), sa.Column('confidence', sa.Float(), nullable=False), sa.Column('status', sa.String(length=50), nullable=False), sa.Column('payload', sa.JSON(), nullable=False), sa.Column('canonical_entity_id', sa.String(length=200), nullable=True), sa.Column('created_at', sa.DateTime(timezone=True), nullable=False), sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('ix_review_items_mention_id', 'review_items', ['mention_id'])
    op.create_index('ix_review_items_source_uap_id', 'review_items', ['source_uap_id'])
    op.create_index('ix_review_items_status', 'review_items', ['status'])
    op.create_table('review_decisions', sa.Column('review_decision_id', sa.Integer(), primary_key=True, autoincrement=True), sa.Column('review_item_id', sa.Integer(), sa.ForeignKey('review_items.review_item_id'), nullable=False), sa.Column('action', sa.String(length=50), nullable=False), sa.Column('reviewer', sa.String(length=100), nullable=False), sa.Column('notes', sa.Text(), nullable=True), sa.Column('remap_target', sa.String(length=200), nullable=True), sa.Column('created_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('ix_review_decisions_review_item_id', 'review_decisions', ['review_item_id'])

def downgrade():
    op.drop_index('ix_review_decisions_review_item_id', table_name='review_decisions')
    op.drop_table('review_decisions')
    op.drop_index('ix_review_items_status', table_name='review_items')
    op.drop_index('ix_review_items_source_uap_id', table_name='review_items')
    op.drop_index('ix_review_items_mention_id', table_name='review_items')
    op.drop_table('review_items')
    op.drop_table('ontology_registry_versions')
