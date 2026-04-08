from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from smap.core.exceptions import DatabaseNotInitializedError
from smap.core.settings import Settings


@dataclass(frozen=True, slots=True)
class DatabaseStatus:
    current_revision: str | None
    head_revision: str
    is_ready: bool
    has_legacy_tables_without_alembic: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_alembic_config(settings: Settings) -> Config:
    config = Config(str(_repo_root() / "alembic.ini"))
    config.set_main_option("script_location", str(_repo_root() / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.db_url)
    config.attributes["smap_url_locked"] = True
    return config


def get_head_revision(settings: Settings) -> str:
    script = ScriptDirectory.from_config(build_alembic_config(settings))
    head_revision = script.get_current_head()
    if head_revision is None:
        raise RuntimeError("Alembic has no head revision configured.")
    return head_revision


def get_current_revision(settings: Settings) -> str | None:
    engine = create_engine(settings.db_url, future=True)
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            if not inspector.has_table("alembic_version"):
                return None
            result = connection.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1")
            ).scalar_one_or_none()
        return result if isinstance(result, str) else None
    finally:
        engine.dispose()


def has_legacy_tables_without_alembic(settings: Settings) -> bool:
    engine = create_engine(settings.db_url, future=True)
    review_tables = {
        "ontology_registry_versions",
        "review_items",
        "review_decisions",
    }
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            has_version_table = inspector.has_table("alembic_version")
            if has_version_table:
                return False
            existing_tables = set(inspector.get_table_names())
        return bool(review_tables & existing_tables)
    finally:
        engine.dispose()


def get_database_status(settings: Settings) -> DatabaseStatus:
    current_revision = get_current_revision(settings)
    head_revision = get_head_revision(settings)
    legacy_without_alembic = has_legacy_tables_without_alembic(settings)
    return DatabaseStatus(
        current_revision=current_revision,
        head_revision=head_revision,
        is_ready=current_revision == head_revision,
        has_legacy_tables_without_alembic=legacy_without_alembic,
    )


def require_database_ready(settings: Settings) -> DatabaseStatus:
    status = get_database_status(settings)
    if status.is_ready:
        return status
    if status.has_legacy_tables_without_alembic:
        raise DatabaseNotInitializedError(
            "Metadata DB predates Alembic. If disposable, delete the local SQLite DB and run "
            "`python -m alembic upgrade head`. Only use `python -m smap.cli.app db-stamp-head` "
            "with `--force` if you have verified the schema matches the current migration."
        )
    raise DatabaseNotInitializedError(
        "Metadata DB is not initialized. Run `python -m alembic upgrade head` or "
        "`python -m smap.cli.app db-upgrade` before using review-backed features."
    )


def upgrade_database(settings: Settings) -> None:
    settings.ensure_directories()
    command.upgrade(build_alembic_config(settings), "head")


def stamp_database_head(settings: Settings, *, force: bool = False) -> None:
    settings.ensure_directories()
    if not force:
        raise DatabaseNotInitializedError(
            "Refusing to stamp metadata DB without `--force`. Stamping only records an Alembic "
            "revision and does not create or upgrade tables."
        )
    status = get_database_status(settings)
    if not status.has_legacy_tables_without_alembic:
        raise DatabaseNotInitializedError(
            "Stamping is only for a legacy metadata DB whose tables already exist and already "
            "match the current schema. For a fresh or drifted DB, run `python -m alembic upgrade head`."
        )
    command.stamp(build_alembic_config(settings), "head")
