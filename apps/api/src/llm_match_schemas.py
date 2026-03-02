from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class ProposedPair(BaseModel):
    item_a_id: int = Field(..., description="ID from contractor (Doc A) item list.")
    item_b_id: Optional[int] = Field(None, description="ID from insurance (Doc B) item list, or null if no match.")
    scope_same: bool = Field(..., description="True if these represent the same scope of work.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the pairing decision.")
    rationale: str = Field(..., description="Short reason for match/unmatch.")

class MatchPlan(BaseModel):
    pairs: list[ProposedPair]
