from sqlalchemy import (
    Column, Integer, Float, String, DateTime,
    JSON, ForeignKey, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


# ── User ─────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String, nullable=False)
    email         = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    results = relationship("AnalysisResult", back_populates="owner", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── AnalysisResult ────────────────────────────────────────────────────────────
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True)

    # Owner
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    owner   = relationship("User", back_populates="results")

    # File info
    filename    = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)

    # Features
    avg_wpm         = Column(Float)
    max_wpm         = Column(Float)
    avg_accuracy    = Column(Float)
    avg_consistency = Column(Float)

    # Analysis
    speed_loss          = Column(Float)
    current_wpm         = Column(Float)
    current_accuracy    = Column(Float)
    current_consistency = Column(Float)
    wpm_variance        = Column(Float)
    fatigue_score       = Column(Float)
    trend               = Column(String)
    fatigue_risk        = Column(String)

    # Predictions
    predicted_future_wpm = Column(Float)
    predicted_change     = Column(Float)

    # Recommendations + raw chart data
    recommendations = Column(JSON)
    wpm_history     = Column(JSON)

    def to_dict(self):
        return {
            "id":          self.id,
            "filename":    self.filename,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "features": {
                "avg_wpm":         self.avg_wpm,
                "max_wpm":         self.max_wpm,
                "avg_accuracy":    self.avg_accuracy,
                "avg_consistency": self.avg_consistency,
            },
            "analysis": {
                "speed_loss":            self.speed_loss,
                "current_wpm":           self.current_wpm,
                "current_accuracy":      self.current_accuracy,
                "current_consistency":   self.current_consistency,
                "wpm_variance":          self.wpm_variance,
                "fatigue_score":         self.fatigue_score,
                "trend":                 self.trend,
                "fatigue_risk":          self.fatigue_risk,
                "predicted_future_wpm":  self.predicted_future_wpm,
                "predicted_change":      self.predicted_change,
            },
            "recommendations": self.recommendations,
            "wpm_history":     self.wpm_history,
        }


import os
from dotenv import load_dotenv

load_dotenv()

# ── Database setup ────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./typing_coach.db")

# SQLAlchemy requires 'postgresql://' instead of 'postgres://' (which Neon sometimes gives)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Only SQLite needs check_same_thread
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create tables (only for new tables; Alembic handles migrations)
Base.metadata.create_all(bind=engine)
