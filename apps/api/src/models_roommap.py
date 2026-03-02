from datetime import datetime
from sqlalchemy import String, DateTime, Integer, Float, Text
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class RoomMap(Base):
    __tablename__ = "room_maps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)

    room_a: Mapped[str] = mapped_column(String, index=True)
    room_b: Mapped[str] = mapped_column(String, index=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    rationale: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
