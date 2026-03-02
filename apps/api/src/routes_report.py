from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .models_matches import Match
from .render_report import render_report_pdf

router = APIRouter(prefix="/runs", tags=["report"])

REPORT_DIR = Path("data/reports")

@router.post("/{run_id}/render")
def render(run_id: str):
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        matches = db.query(Match).filter(Match.run_id == run_id).all()
        if not matches:
            raise HTTPException(status_code=400, detail="no matches; run /match first")

        # Pull descriptions/amounts for referenced items
        ids_a = [m.item_a_id for m in matches if m.item_a_id is not None]
        ids_b = [m.item_b_id for m in matches if m.item_b_id is not None]

        a_items = {x.id: x for x in db.query(LineItem).filter(LineItem.id.in_(ids_a)).all()} if ids_a else {}
        b_items = {x.id: x for x in db.query(LineItem).filter(LineItem.id.in_(ids_b)).all()} if ids_b else {}

        rows = []
        for m in matches:
            a = a_items.get(m.item_a_id) if m.item_a_id is not None else None
            b = b_items.get(m.item_b_id) if m.item_b_id is not None else None
            rows.append({
                "room_a": m.room_a,
                "room_b": m.room_b,
                "status": m.status,
                "rationale": m.rationale,
                "a_desc": a.description if a else "",
                "b_desc": b.description if b else "",
                "a_amt": a.amount if a else None,
                "b_amt": b.amount if b else None,
            })

        summary = {
            "green": sum(1 for m in matches if m.status == "green"),
            "orange": sum(1 for m in matches if m.status == "orange"),
            "blue": sum(1 for m in matches if m.status == "blue"),
            "total_a": sum((a_items[m.item_a_id].amount or 0.0) for m in matches if m.status != "blue" and m.item_a_id in a_items),
            "total_b": sum((b_items[m.item_b_id].amount or 0.0) for m in matches if m.status != "blue" and m.item_b_id in b_items),
        }

        out_path = REPORT_DIR / f"{run_id}.pdf"
        render_report_pdf(str(out_path), run_id=run_id, rows=rows, summary=summary)
        return {"run_id": run_id, "report_path": str(out_path)}

@router.get("/{run_id}/report")
def download(run_id: str):
    path = REPORT_DIR / f"{run_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not found; run POST /render first")
    return FileResponse(path, media_type="application/pdf", filename=f"ciridae_report_{run_id}.pdf")
