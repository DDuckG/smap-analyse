from dataclasses import dataclass
from pathlib import Path
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text
from smap.core.exceptions import DatabaseNotInitializedError

@dataclass(frozen=True, slots=True)
class DatabaseStatus:
    current_revision: str | None
    head_revision: str
    is_ready: bool
    has_legacy_tables_without_alembic: bool

def _repo_root():
    return Path(__file__).resolve().parents[3]

def build_alembic_config(settings):
    config = Config(str(_repo_root() / 'alembic.ini'))
    config.set_main_option('script_location', str(_repo_root() / 'alembic'))
    config.set_main_option('sqlalchemy.url', settings.db_url)
    config.attributes['smap_url_locked'] = True
    return config

def get_head_revision(settings):
    script = ScriptDirectory.from_config(build_alembic_config(settings))
    head_revision = script.get_current_head()
    if head_revision is None:
        raise RuntimeError('Alembic has no head revision configured.')
    return head_revision

def get_current_revision(settings):
    engine = create_engine(settings.db_url, future=True)
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            if not inspector.has_table('alembic_version'):
                return None
            result = connection.execute(text('SELECT version_num FROM alembic_version LIMIT 1')).scalar_one_or_none()
        return result if isinstance(result, str) else None
    finally:
        engine.dispose()

def has_legacy_tables_without_alembic(settings):
    engine = create_engine(settings.db_url, future=True)
    review_tables = {'ontology_registry_versions', 'review_items', 'review_decisions'}
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            if inspector.has_table('alembic_version'):
                return False
            existing_tables = set(inspector.get_table_names())
        return bool(review_tables & existing_tables)
    finally:
        engine.dispose()

def get_database_status(settings):
    current_revision = get_current_revision(settings)
    head_revision = get_head_revision(settings)
    legacy_without_alembic = has_legacy_tables_without_alembic(settings)
    return DatabaseStatus(current_revision=current_revision, head_revision=head_revision, is_ready=current_revision == head_revision, has_legacy_tables_without_alembic=legacy_without_alembic)

def require_database_ready(settings):
    status = get_database_status(settings)
    if status.is_ready:
        return status
    if status.has_legacy_tables_without_alembic:
        raise DatabaseNotInitializedError('Metadata DB predates Alembic. Recreate the local SQLite file or run `python -m alembic upgrade head` after verifying the current schema.')
    raise DatabaseNotInitializedError('Metadata DB is not initialized. Run `python -m alembic upgrade head` or `smap db-upgrade` first.')

def upgrade_database(settings):
    settings.ensure_directories()
    command.upgrade(build_alembic_config(settings), 'head')
