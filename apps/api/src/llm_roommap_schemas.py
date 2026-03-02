from __future__ import annotations
from pydantic import BaseModel, Field

class RoomLink(BaseModel):
    room_a: str = Field(..., description="Exact room string from contractor (Doc A) list.")
    room_b: str = Field(..., description="Exact room string from insurance (Doc B) list.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this mapping.")
    rationale: str = Field(..., description="Short reason: rename, same area, split/merge, etc.")

class RoomMapResult(BaseModel):
    links: list[RoomLink]
