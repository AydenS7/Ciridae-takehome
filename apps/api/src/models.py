"""SQLAlchemy model for uploaded run metadata and source proposal paths."""

import uuid
from datetime import datetime
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    proposal_a_path: Mapped[str] = mapped_column(String)
    proposal_b_path: Mapped[str] = mapped_column(String)
