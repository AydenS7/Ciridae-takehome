"""Pydantic response schemas for all API endpoints."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    run_id: str
    proposal_a_path: str
    proposal_b_path: str


# ── Extract ───────────────────────────────────────────────────────────────────

class ExtractResponse(BaseModel):
    run_id: str
    extracted: dict[str, int]
    metrics: dict[str, Any]


class LineItemResponse(BaseModel):
    id: int
    doc: str
    page: int
    room: str
    description: str
    amount: float | None


# ── Room mapping ──────────────────────────────────────────────────────────────

class RoomMapResponse(BaseModel):
    run_id: str
    rooms_a: int
    rooms_b: int
    links: int
    metrics: dict[str, Any]
    room_groups: list[Any]


class RoomLinkResponse(BaseModel):
    room_a: str
    room_b: str
    confidence: float
    rationale: str | None


# ── Matching ──────────────────────────────────────────────────────────────────

class MatchResponse(BaseModel):
    run_id: str
    matches_inserted: int
    first_pass_model: str
    second_pass_models: list[str]
    first_pass_total_evaluated: int
    first_pass_uncertain_count: int
    second_pass_reviewed_count: int
    second_pass_rooms_invoked: int
    nugget_count: int
    status_counts: dict[str, int]
    coverage_audit: dict[str, Any]
    llm_telemetry: dict[str, Any]
    elapsed_ms: int
    matching_mode: str
    room_group_component_count: int
    room_group_room_a_coverage: int


class MatchItemResponse(BaseModel):
    id: int
    run_id: str
    status: str
    room_a: str | None
    room_b: str | None
    item_a_id: int | None
    item_b_id: int | None
    similarity: float | None
    rationale: str | None


# ── Render ────────────────────────────────────────────────────────────────────

class RenderResponse(BaseModel):
    run_id: str
    report_path: str
    metrics: dict[str, Any]
