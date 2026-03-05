from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Doc = Literal["A", "B"]

class ExtractedItem(BaseModel):
    # "room" should be the closest room/area/section heading for the item
    room: str = Field(..., description="Room/area/section name. Use best guess; never empty.")
    description: str = Field(..., description="Line item description (no page headers/footers).")
    quantity: Optional[float] = Field(None, description="Quantity if present, else null.")
    unit: Optional[str] = Field(None, description="Unit like EA, SF, LF, HR, etc if present, else null.")
    unit_price: Optional[float] = Field(None, description="Unit price (cost per unit) if present, else null.")
    total: Optional[float] = Field(None, description="Total cost for the line item if present, else null.")
    # Helpful for debugging / traceability
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence the row is a real estimate line item.")

class ExtractPageResult(BaseModel):
    doc: Doc
    page: int
    items: list[ExtractedItem]
