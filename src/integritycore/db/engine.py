"""SQLAlchemy engine and session factory for IntegrityCore."""
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from integritycore.db.models import Base

# Default: SQLite in ~/.integritycore/
_default_db = Path.home() / ".integritycore" / "integritycore.db"
_default_db.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_default_db}")

# connect_args only valid for SQLite
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they do not exist. Migrate existing jobs table if needed."""
    from integritycore.db.models import GoldQuery  # noqa: F401 - ensure GoldQuery is registered
    Base.metadata.create_all(bind=engine)
    # Migration: add selected_source_table, selected_target_table to existing jobs
    from sqlalchemy import text
    for col in ("selected_source_table", "selected_target_table"):
        try:
            with engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {col} VARCHAR DEFAULT ''"))
                conn.commit()
        except Exception:
            pass  # Column may already exist

    # Migration: add lineage columns to job_runs
    for col, col_type in [
        ("llm_model_version", "TEXT"),
        ("input_ddl_context", "TEXT"),
        ("source_target_map", "TEXT"),
        ("logic_verification_result", "TEXT"),
    ]:
        try:
            with engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE job_runs ADD COLUMN {col} {col_type}"))
                conn.commit()
        except Exception:
            pass  # Column may already exist


@contextmanager
def get_db() -> Session:
    """Context-manager session — auto-commit on success, rollback on error."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
