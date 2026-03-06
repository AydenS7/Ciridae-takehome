"""Streaming endpoint that runs extract, room-map, match, and report in sequence."""

from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .routes_extract import extract_run
from .routes_roommap import map_rooms
from .routes_match import match_run
from .routes_report import render

router = APIRouter(prefix="/runs", tags=["pipeline"])


def _event(step: str, status: str, msg: str = "", data: dict | None = None) -> str:
    payload: dict = {"step": step, "status": status}
    if msg:
        payload["msg"] = msg
    if data:
        payload["data"] = data
    return f"data: {json.dumps(payload)}\n\n"


def _pipeline_generator(run_id: str):
    # Step 1: Extract
    yield _event("extract", "start", "Extracting line items from both PDFs (vision + text)…")
    try:
        extract_result = extract_run(run_id)
        a_count = extract_result.get("extracted", {}).get("A", 0)
        b_count = extract_result.get("extracted", {}).get("B", 0)
        vision_a = extract_result.get("metrics", {}).get("docs", {}).get("A", {}).get("vision_pages", 0)
        vision_b = extract_result.get("metrics", {}).get("docs", {}).get("B", {}).get("vision_pages", 0)
        yield _event(
            "extract", "done",
            f"Extracted {a_count} JDR items, {b_count} insurance items "
            f"(vision pages: A={vision_a}, B={vision_b}).",
            data=extract_result,
        )
    except HTTPException as e:
        yield _event("extract", "error", str(e.detail))
        return
    except Exception as e:
        yield _event("extract", "error", str(e))
        return

    # Step 2: Map rooms
    yield _event("map_rooms", "start", "Mapping rooms between documents…")
    try:
        map_result = map_rooms(run_id)
        links = map_result.get("links", 0)
        model = map_result.get("metrics", {}).get("model_used", "unknown")
        yield _event(
            "map_rooms", "done",
            f"Room mapping complete: {links} link(s) found (model: {model}).",
            data=map_result,
        )
    except HTTPException as e:
        yield _event("map_rooms", "error", str(e.detail))
        return
    except Exception as e:
        yield _event("map_rooms", "error", str(e))
        return

    # Step 3: Match
    yield _event("match", "start", "Matching line items (first-pass + reviewer second-pass)…")
    try:
        match_result = match_run(run_id)
        green = (match_result.get("status_counts") or {}).get("green", 0)
        orange = (match_result.get("status_counts") or {}).get("orange", 0)
        blue = (match_result.get("status_counts") or {}).get("blue", 0)
        nuggets = match_result.get("nugget_count", 0)
        critical = (match_result.get("coverage_audit") or {}).get("critical_blue_count", 0)
        yield _event(
            "match", "done",
            f"Matching complete: {green} green, {orange} orange, {blue} blue, {nuggets} nuggets, {critical} critical.",
            data=match_result,
        )
    except HTTPException as e:
        yield _event("match", "error", str(e.detail))
        return
    except Exception as e:
        yield _event("match", "error", str(e))
        return

    # Step 4: Render
    yield _event("render", "start", "Rendering annotated PDF report…")
    try:
        render_result = render(run_id)
        stats = render_result.get("metrics", {}).get("render_stats", {})
        highlights = stats.get("highlights_added", 0)
        unlocated = stats.get("unlocated_notes_added", 0)
        summary_page = stats.get("summary_page_appended", False)
        yield _event(
            "render", "done",
            f"Report ready: {highlights} highlights, {unlocated} unlocated notes. "
            f"Summary page: {'yes' if summary_page else 'no'}.",
            data=render_result,
        )
    except HTTPException as e:
        yield _event("render", "error", str(e.detail))
        return
    except Exception as e:
        yield _event("render", "error", str(e))
        return

    yield _event("done", "done", "Pipeline complete.", data={"run_id": run_id})


@router.get("/{run_id}/pipeline/stream")
def pipeline_stream(run_id: str):
    """Stream pipeline progress as Server-Sent Events (SSE)."""
    return StreamingResponse(
        _pipeline_generator(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
