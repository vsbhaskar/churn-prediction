import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import JSON, Boolean, DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger(__name__)

load_dotenv()  # reads the .env file into environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# create_engine does not connect yet; pool_pre_ping checks a connection
# is still alive before reusing it
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)  # stored as UTC


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One row per prediction the API has made."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    risk_score: Mapped[float] = mapped_column(Float)
    flag_for_offer: Mapped[bool] = mapped_column(Boolean)
    threshold: Mapped[float] = mapped_column(Float)
    features: Mapped[dict] = mapped_column(JSON)       # the customer details sent in
    top_reasons: Mapped[list] = mapped_column(JSON)    # the SHAP reasons we returned


def init_db():
    """Create the table if it does not exist yet."""
    if engine is None:
        logger.warning("DATABASE_URL not set: prediction logging is disabled")
        return
    try:
        Base.metadata.create_all(engine)
    except Exception:
        logger.exception("Could not initialise the database; logging may fail")


def log_prediction(*, model_version, risk_score, flag_for_offer, threshold,
                   features, top_reasons):
    """Save one prediction. Never raises: logging must not break the API."""
    if engine is None:
        return
    try:
        with Session(engine) as session:
            session.add(Prediction(
                model_version=model_version,
                risk_score=risk_score,
                flag_for_offer=flag_for_offer,
                threshold=threshold,
                features=features,
                top_reasons=top_reasons,
            ))
            session.commit()
    except Exception:
        logger.exception("Failed to log prediction")