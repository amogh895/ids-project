"""Database models for the NIDS alert system.

Uses SQLAlchemy ORM so swapping SQLite for Postgres is a one-line change
in the connection string.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "nids.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


class Alert(Base):
    """Single alert row — one per aggregated alert group."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # 5-tuple identification
    src_ip = Column(String(45), nullable=False, index=True)
    dst_ip = Column(String(45), nullable=False)
    src_port = Column(Integer, nullable=True)
    dst_port = Column(Integer, nullable=True, index=True)
    protocol = Column(Integer, nullable=True)

    # Model predictions
    predicted_class = Column(String(32), nullable=False, index=True)
    supervised_score = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String(16), nullable=False, index=True)
    alert_reason = Column(Text, nullable=True)

    # Aggregation
    hit_count = Column(Integer, default=1)
    agg_key = Column(String(128), nullable=True, index=True)

    # Model version tracking
    model_version = Column(String(64), nullable=True)

    # SHAP explanation (lazily computed, stored as JSON string)
    shap_explanation = Column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "predicted_class": self.predicted_class,
            "supervised_score": round(self.supervised_score, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity,
            "alert_reason": self.alert_reason,
            "hit_count": self.hit_count,
            "model_version": self.model_version,
            "shap_explanation": self.shap_explanation,
        }


# ─── Engine & Session ─────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """FastAPI dependency: yield a DB session, auto-close on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
