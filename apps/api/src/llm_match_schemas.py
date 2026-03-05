from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class ProposedPair(BaseModel):
    item_a_id: int = Field(..., description="ID from contractor (Doc A) item list.")
    item_b_id: Optional[int] = Field(None, description="ID from insurance (Doc B) item list, or null if no match.")
    scope_same: bool = Field(
        ...,
        description=(
            "True if both items represent the same general type of work, even if worded differently. "
            "When item_b_id is not null, default to scope_same=true unless you have specific reason to doubt scope alignment. "
            "Examples: 'demo and haul' and 'debris removal' are the same scope. "
            "'Replace carpet' and 'carpet installation' are the same scope. "
            "Set scope_same=false only when you matched items but are genuinely uncertain whether the underlying scopes overlap."
        ),
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the pairing decision.")
    rationale: str = Field(..., description="Short reason for match/unmatch.")
    critical_blue: bool = Field(
        False,
        description=(
            "Only relevant when item_b_id is null (JDR-only). "
            "Set to true if this line item is high-priority scope the insurer should cover: "
            "safety testing, permits, code compliance, environmental hazards (mold, asbestos, lead), "
            "engineering reports, inspections, or items with significant liability implications."
        ),
    )

class MatchPlan(BaseModel):
    pairs: list[ProposedPair]
