from datetime import datetime
from sqlalchemy import String, DateTime, Integer, Float, Text
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class LineItem(Base):
    __tablename__ = "line_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    run_id: Mapped[str] = mapped_column(String, index=True)
    doc: Mapped[str] = mapped_column(String)  # "A" or "B"
    page: Mapped[int] = mapped_column(Integer)

    room: Mapped[str] = mapped_column(String, default="(unknown)")
    description: Mapped[str] = mapped_column(Text)

    amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
