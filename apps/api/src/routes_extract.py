from fastapi import APIRouter, HTTPException
from sqlalchemy import delete

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .extract_pdf import extract_lines

router = APIRouter(prefix="/runs", tags=["runs"])

@router.post("/{run_id}/extract")
def extract_run(run_id: str):
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        # Idempotent: clear previous extracted items for this run
        db.execute(delete(LineItem).where(LineItem.run_id == run_id))
        db.commit()

        # Extract A
        a_lines = extract_lines(run.proposal_a_path)
        for ln in a_lines:
            db.add(LineItem(
                run_id=run_id, doc="A", page=ln.page, room=ln.room,
                description=ln.description, amount=ln.amount
            ))

        # Extract B
        b_lines = extract_lines(run.proposal_b_path)
        for ln in b_lines:
            db.add(LineItem(
                run_id=run_id, doc="B", page=ln.page, room=ln.room,
                description=ln.description, amount=ln.amount
            ))

        db.commit()

        return {
            "run_id": run_id,
            "extracted": {"A": len(a_lines), "B": len(b_lines)},
        }

@router.get("/{run_id}/items")
def list_items(run_id: str, doc: str | None = None, limit: int = 200):
    with SessionLocal() as db:
        q = db.query(LineItem).filter(LineItem.run_id == run_id)
        if doc:
            q = q.filter(LineItem.doc == doc)
        items = q.order_by(LineItem.doc, LineItem.page, LineItem.id).limit(limit).all()

        return [
            {
                "id": it.id,
                "doc": it.doc,
                "page": it.page,
                "room": it.room,
                "description": it.description,
                "amount": it.amount,
            }
            for it in items
        ]
