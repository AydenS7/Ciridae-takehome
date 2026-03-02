from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, func

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .models_roommap import RoomMap
from .room_mapping import map_rooms_via_llm

router = APIRouter(prefix="/runs", tags=["room-mapping"])

@router.post("/{run_id}/map-rooms")
def map_rooms(run_id: str):
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        # Must have extracted items first
        n_items = db.query(func.count(LineItem.id)).filter(LineItem.run_id == run_id).scalar() or 0
        if n_items == 0:
            raise HTTPException(status_code=400, detail="no extracted items; run /extract first")

        rooms_a = [r[0] for r in db.query(LineItem.room).filter(LineItem.run_id == run_id, LineItem.doc == "A").distinct().all()]
        rooms_b = [r[0] for r in db.query(LineItem.room).filter(LineItem.run_id == run_id, LineItem.doc == "B").distinct().all()]

        # Clear previous mapping (idempotent)
        db.execute(delete(RoomMap).where(RoomMap.run_id == run_id))
        db.commit()

        llm_result = map_rooms_via_llm(rooms_a=rooms_a, rooms_b=rooms_b)

        # Persist
        inserted = 0
        seen = set()
        for link in llm_result.links:
            key = (link.room_a, link.room_b)
            if key in seen:
                continue
            seen.add(key)
            db.add(RoomMap(
                run_id=run_id,
                room_a=link.room_a,
                room_b=link.room_b,
                confidence=link.confidence,
                rationale=link.rationale,
            ))
            inserted += 1

        db.commit()
        return {"run_id": run_id, "rooms_a": len(rooms_a), "rooms_b": len(rooms_b), "links": inserted}

@router.get("/{run_id}/map-rooms")
def get_room_map(run_id: str, min_confidence: float = 0.6):
    with SessionLocal() as db:
        links = (
            db.query(RoomMap)
            .filter(RoomMap.run_id == run_id, RoomMap.confidence >= min_confidence)
            .order_by(RoomMap.confidence.desc())
            .all()
        )
        return [
            {
                "room_a": l.room_a,
                "room_b": l.room_b,
                "confidence": l.confidence,
                "rationale": l.rationale,
            }
            for l in links
        ]
