from fastapi import APIRouter, HTTPException
from sqlalchemy import delete

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .extract_pdf_llm import extract_pdf_via_llm

router = APIRouter(prefix="/runs", tags=["runs"])

@router.post("/{run_id}/extract")
def extract_run(run_id: str):
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        db.execute(delete(LineItem).where(LineItem.run_id == run_id))
        db.commit()

        a_pages = extract_pdf_via_llm(run.proposal_a_path, doc="A")
        b_pages = extract_pdf_via_llm(run.proposal_b_path, doc="B")

        def persist(doc_pages):
            n = 0
            seen = set()
            for page_result in doc_pages:
                for it in page_result.items:
                    key = (
                        page_result.doc,
                        page_result.page,
                        it.room.strip().lower(),
                        it.description.strip().lower(),
                        round(it.total or 0.0, 2) if it.total is not None else None,
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    db.add(LineItem(
                        run_id=run_id,
                        doc=page_result.doc,
                        page=page_result.page,
                        room=it.room,
                        description=it.description,
                        amount=it.total,
                    ))
                    n += 1
            return n

        n_a = persist(a_pages)
        n_b = persist(b_pages)
        db.commit()

        return {"run_id": run_id, "extracted": {"A": n_a, "B": n_b}}

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
