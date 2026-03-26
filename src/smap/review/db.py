from __future__ import annotations
from collections.abc import Iterator
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from smap.core.settings import Settings
from smap.review.migrations import require_database_ready

def build_engine(settings):
    return create_engine(settings.db_url, future=True)

def build_session_factory(settings, *, require_ready=True):
    if require_ready:
        require_database_ready(settings)
    return sessionmaker(bind=build_engine(settings), expire_on_commit=False, future=True)

@contextmanager
def session_scope(settings, *, require_ready=True):
    engine = build_engine(settings)
    if require_ready:
        from smap.review.migrations import require_database_ready
        require_database_ready(settings)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()
