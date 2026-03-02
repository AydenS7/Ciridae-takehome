from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, func

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .models_roommap import RoomMap
from .models_matches import Match
from .matching_llm import propose_matches_for_room

router = APIRouter(prefix="/runs", tags=["matching"])

def _pct_diff(a: float, b: float) -> float:
    # avoid div by zero; treat as infinite diff if either missing or zero-ish
    if a is None or b is None:
        return float("inf")
    denom = max(abs(b), 1e-9)
    return abs(a - b) / denom

@router.post("/{run_id}/match")
def match_run(run_id: str, min_room_confidence: float = 0.6, min_pair_confidence: float = 0.6):
    """
    LLM-only pairing per mapped room, deterministic classification:
    - blue: A-only (no B match)
    - green: scope_same AND amount within ±2%
    - orange: matched but scope differs OR amount differs >2%
    """
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        # must have extracted items
        n_items = db.query(func.count(LineItem.id)).filter(LineItem.run_id == run_id).scalar() or 0
        if n_items == 0:
            raise HTTPException(status_code=400, detail="no extracted items; run /extract first")

        # must have room mapping
        room_links = (
            db.query(RoomMap)
            .filter(RoomMap.run_id == run_id, RoomMap.confidence >= min_room_confidence)
            .all()
        )
        if not room_links:
            raise HTTPException(status_code=400, detail="no room mapping; run /map-rooms first")

        # clear previous matches
        db.execute(delete(Match).where(Match.run_id == run_id))
        db.commit()

        inserted = 0

        for link in room_links:
            room_a = link.room_a
            room_b = link.room_b

            a_items = (
                db.query(LineItem)
                .filter(LineItem.run_id == run_id, LineItem.doc == "A", LineItem.room == room_a)
                .order_by(LineItem.page, LineItem.id)
                .all()
            )
            b_items = (
                db.query(LineItem)
                .filter(LineItem.run_id == run_id, LineItem.doc == "B", LineItem.room == room_b)
                .order_by(LineItem.page, LineItem.id)
                .all()
            )

            if not a_items:
                continue

            # Convert to dicts for prompt
            a_dicts = [{"id": x.id, "room": x.room, "description": x.description, "amount": x.amount} for x in a_items]
            b_dicts = [{"id": x.id, "room": x.room, "description": x.description, "amount": x.amount} for x in b_items]

            plan = propose_matches_for_room(room_a, room_b, a_dicts, b_dicts)

            # Enforce one-to-one on B ids even if LLM violates it
            used_b: set[int] = set()
            b_by_id = {x.id: x for x in b_items}
            a_by_id = {x.id: x for x in a_items}

            for p in plan.pairs:
                if p.item_a_id not in a_by_id:
                    continue  # ignore bad ids
                if p.confidence < min_pair_confidence:
                    # treat as unmatched
                    p_item_b = None
                else:
                    p_item_b = p.item_b_id

                if p_item_b is not None:
                    if p_item_b not in b_by_id:
                        p_item_b = None
                    elif p_item_b in used_b:
                        p_item_b = None
                    else:
                        used_b.add(p_item_b)

                a_amt = a_by_id[p.item_a_id].amount
                b_amt = b_by_id[p_item_b].amount if p_item_b is not None else None

                if p_item_b is None:
                    status = "blue"
                else:
                    within_2pct = (_pct_diff(a_amt, b_amt) <= 0.02) if (a_amt is not None and b_amt is not None) else False
                    if p.scope_same and within_2pct:
                        status = "green"
                    else:
                        status = "orange"

                db.add(Match(
                    run_id=run_id,
                    room_a=room_a,
                    room_b=room_b,
                    item_a_id=p.item_a_id,
                    item_b_id=p_item_b,
                    status=status,
                    similarity=p.confidence,
                    rationale=p.rationale,
                ))
                inserted += 1

        db.commit()
        return {"run_id": run_id, "matches_inserted": inserted}

@router.get("/{run_id}/matches")
def list_matches(run_id: str, status: str | None = None, limit: int = 200):
    with SessionLocal() as db:
        q = db.query(Match).filter(Match.run_id == run_id)
        if status:
            q = q.filter(Match.status == status)

        rows = q.order_by(Match.id).limit(limit).all()
        return [
            {
                "room_a": r.room_a,
                "room_b": r.room_b,
                "item_a_id": r.item_a_id,
                "item_b_id": r.item_b_id,
                "status": r.status,
                "confidence": r.similarity,
                "rationale": r.rationale,
            }
            for r in rows
        ]
