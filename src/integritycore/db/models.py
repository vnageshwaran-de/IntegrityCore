"""SQLAlchemy ORM models for IntegrityCore job store."""
import uuid
import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, Enum
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


def _uuid():
    return str(uuid.uuid4())

def _now():
    return datetime.datetime.utcnow()


class JobStatus(str, enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """An ETL job definition — source, target, prompt, and schedule."""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")

    # Connections (names from connections.json)
    source_conn_id = Column(String, nullable=False)
    target_conn_id = Column(String, nullable=False)

    # LLM prompt
    prompt = Column(Text, nullable=False)

    # Schedule — cron expression or blank for manual-only
    schedule_cron = Column(String, default="")   # e.g. "0 * * * *"
    schedule_label = Column(String, default="")  # e.g. "Every hour"

    status = Column(String, default=JobStatus.ACTIVE)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    # Relationships
    runs = relationship("JobRun", back_populates="job", order_by="desc(JobRun.started_at)", cascade="all, delete-orphan")

    def to_dict(self):
        last_run = self.runs[0] if self.runs else None
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "source_conn_id": self.source_conn_id,
            "target_conn_id": self.target_conn_id,
            "prompt": self.prompt,
            "schedule_cron": self.schedule_cron,
            "schedule_label": self.schedule_label,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_run": last_run.to_dict() if last_run else None,
            "run_count": len(self.runs),
        }


class JobRun(Base):
    """A single execution instance of a Job."""
    __tablename__ = "job_runs"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)

    status = Column(String, default=RunStatus.PENDING)
    triggered_by = Column(String, default="manual")  # 'manual' | 'schedule'

    started_at = Column(DateTime, default=_now)
    finished_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    rows_processed = Column(Integer, default=0)
    logs = Column(Text, default="")
    error_msg = Column(Text, default="")

    # Generated SQL for auditing
    generated_sql = Column(Text, default="")
    verified = Column(Boolean, default=False)

    job = relationship("Job", back_populates="runs")

    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "status": self.status,
            "triggered_by": self.triggered_by,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "rows_processed": self.rows_processed,
            "logs": self.logs,
            "error_msg": self.error_msg,
            "generated_sql": self.generated_sql,
            "verified": self.verified,
        }
