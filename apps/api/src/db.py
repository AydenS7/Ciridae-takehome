"""Database engine/session setup plus startup-safe schema initialization."""

from pathlib import Path
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(autoflush=False, autocommit=False, future=True)
SessionLocal.configure(bind=engine)

SQLITE_FALLBACK_URL = "sqlite:///./data/local-dev.db"


def _can_use_sqlite_fallback(database_url: str) -> bool:
    # Keep fallback limited to local dev so production-like configs still fail loudly.
    return database_url.startswith("postgresql") and (
        "@localhost" in database_url or "@127.0.0.1" in database_url
    )


def _switch_to_sqlite_fallback() -> None:
    global engine

    Path("data").mkdir(parents=True, exist_ok=True)
    engine = create_engine(SQLITE_FALLBACK_URL, future=True)
    SessionLocal.configure(bind=engine)


def _run_migrations(eng) -> None:
    """Add new columns to existing tables without dropping data."""
    try:
        insp = inspect(eng)
        if "line_items" in insp.get_table_names():
            existing = {col["name"] for col in insp.get_columns("line_items")}
            new_cols = [
                ("quantity", "REAL"),
                ("unit", "TEXT"),
                ("unit_price", "REAL"),
            ]
            with eng.begin() as conn:
                for col_name, col_type in new_cols:
                    if col_name not in existing:
                        conn.execute(text(f"ALTER TABLE line_items ADD COLUMN {col_name} {col_type}"))
                        logger.info("Migration: added column line_items.%s", col_name)
    except Exception as exc:
        logger.warning("Migration check failed (non-fatal): %s", exc)


def init_db() -> None:
    # Ensure models are imported so Base.metadata includes them.
    from . import models           # Run
    from . import models_items     # LineItem
    from . import models_roommap   # RoomMap
    from . import models_matches   # Match

    try:
        Base.metadata.create_all(bind=engine)
        _run_migrations(engine)
    except OperationalError:
        if not _can_use_sqlite_fallback(settings.database_url):
            raise
        logger.warning(
            "Primary database unavailable at %s; falling back to %s for local dev.",
            settings.database_url,
            SQLITE_FALLBACK_URL,
        )
        _switch_to_sqlite_fallback()
        Base.metadata.create_all(bind=engine)
        _run_migrations(engine)
