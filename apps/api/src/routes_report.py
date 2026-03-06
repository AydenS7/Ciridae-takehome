"""Endpoints for building and serving the annotated reconciliation PDF report."""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func
from time import perf_counter

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .models_matches import Match
from .render_report import render_report_pdf

router = APIRouter(prefix="/runs", tags=["report"])

REPORT_DIR = Path("data/reports")

@router.post("/{run_id}/render")
def render(run_id: str):
    t0 = perf_counter()
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        matches = db.query(Match).filter(Match.run_id == run_id).order_by(Match.id).all()
        if not matches:
            raise HTTPException(status_code=400, detail="no matches; run /match first")

        # Pull descriptions/amounts for referenced items
        ids_a = [m.item_a_id for m in matches if m.item_a_id is not None]
        ids_b = [m.item_b_id for m in matches if m.item_b_id is not None]

        a_items = {x.id: x for x in db.query(LineItem).filter(LineItem.id.in_(ids_a)).all()} if ids_a else {}
        b_items = {x.id: x for x in db.query(LineItem).filter(LineItem.id.in_(ids_b)).all()} if ids_b else {}
        a_room_first_page = {
            room: page
            for room, page in (
                db.query(LineItem.room, func.min(LineItem.page))
                .filter(LineItem.run_id == run_id, LineItem.doc == "A")
                .group_by(LineItem.room)
                .all()
            )
        }

        rows = []
        for m in matches:
            a = a_items.get(m.item_a_id) if m.item_a_id is not None else None
            b = b_items.get(m.item_b_id) if m.item_b_id is not None else None
            a_page = a.page if a else a_room_first_page.get(m.room_a)
            if a_page is None:
                a_page = 1
            rows.append({
                "a_item_id": m.item_a_id,
                "b_item_id": m.item_b_id,
                "room_a": m.room_a,
                "room_b": m.room_b,
                "status": m.status,
                "rationale": m.rationale,
                "a_desc": a.description if a else "",
                "b_desc": b.description if b else "",
                "a_amt": a.amount if a else None,
                "b_amt": b.amount if b else None,
                "a_qty": a.quantity if a else None,
                "b_qty": b.quantity if b else None,
                "a_unit": a.unit if a else None,
                "b_unit": b.unit if b else None,
                "a_unit_price": a.unit_price if a else None,
                "b_unit_price": b.unit_price if b else None,
                "a_page": a_page,
                "b_page": b.page if b else None,
                "critical_blue": bool("[CRITICAL_BLUE]" in (m.rationale or "")),
            })

        summary = {
            "green": sum(1 for m in matches if m.status == "green"),
            "orange": sum(1 for m in matches if m.status == "orange"),
            "blue": sum(1 for m in matches if m.status == "blue"),
            "nugget": sum(1 for m in matches if m.status == "nugget"),
            "total_a": sum((a_items[m.item_a_id].amount or 0.0) for m in matches if m.status in {"green", "orange"} and m.item_a_id in a_items),
            "total_b": sum((b_items[m.item_b_id].amount or 0.0) for m in matches if m.status in {"green", "orange"} and m.item_b_id in b_items),
        }

        out_path = REPORT_DIR / f"{run_id}.pdf"
        _, render_stats = render_report_pdf(
            str(out_path),
            source_pdf_path=run.proposal_a_path,
            run_id=run_id,
            rows=rows,
            summary=summary,
        )
        elapsed_ms = int((perf_counter() - t0) * 1000)
        return {
            "run_id": run_id,
            "report_path": str(out_path),
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "render_stats": render_stats,
            },
        }

@router.get("/{run_id}/report")
def download(run_id: str):
    path = REPORT_DIR / f"{run_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not found; run POST /render first")
    return FileResponse(path, media_type="application/pdf", filename=f"ciridae_report_{run_id}.pdf")
