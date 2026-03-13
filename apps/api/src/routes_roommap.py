"""Endpoints for generating and persisting room-to-room mappings per run."""

import re
from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, func, select
from time import perf_counter

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .models_roommap import RoomMap
from .room_mapping import build_room_groups_from_links, map_rooms_via_llm
from .llm_roommap_schemas import RoomLink
from .schemas import RoomLinkResponse, RoomMapResponse

router = APIRouter(prefix="/runs", tags=["room-mapping"])


def _desc_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(tok) > 2
    }


@router.post("/{run_id}/map-rooms", response_model=RoomMapResponse)
def map_rooms(run_id: str):
    t0 = perf_counter()
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        # Must have extracted items first
        n_items = db.scalar(select(func.count(LineItem.id)).where(LineItem.run_id == run_id)) or 0
        if n_items == 0:
            raise HTTPException(status_code=400, detail="no extracted items; run /extract first")

        rooms_a = list(db.scalars(select(LineItem.room).where(LineItem.run_id == run_id, LineItem.doc == "A").distinct()))
        rooms_b = list(db.scalars(select(LineItem.room).where(LineItem.run_id == run_id, LineItem.doc == "B").distinct()))

        room_profile_tokens_a: dict[str, set[str]] = {}
        room_profile_tokens_b: dict[str, set[str]] = {}
        for room, desc, doc in db.execute(
            select(LineItem.room, LineItem.description, LineItem.doc)
            .where(LineItem.run_id == run_id)
        ):
            if doc == "A":
                room_profile_tokens_a.setdefault(room, set()).update(_desc_tokens(desc or ""))
            elif doc == "B":
                room_profile_tokens_b.setdefault(room, set()).update(_desc_tokens(desc or ""))

        # Clear previous mapping (idempotent)
        db.execute(delete(RoomMap).where(RoomMap.run_id == run_id))
        db.commit()

        llm_result, telemetry = map_rooms_via_llm(
            rooms_a=rooms_a,
            rooms_b=rooms_b,
            room_profile_tokens_a=room_profile_tokens_a,
            room_profile_tokens_b=room_profile_tokens_b,
        )

        # Persist
        inserted = 0
        seen = set()
        persisted_links: list[RoomLink] = []
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
            persisted_links.append(link)
            inserted += 1

        db.commit()
        elapsed_ms = int((perf_counter() - t0) * 1000)

        room_groups = build_room_groups_from_links(
            rooms_a=rooms_a,
            rooms_b=rooms_b,
            links=persisted_links,
            min_confidence=0.48,
        )

        return {
            "run_id": run_id,
            "rooms_a": len(rooms_a),
            "rooms_b": len(rooms_b),
            "links": inserted,
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "model_used": telemetry.get("model_used"),
                "attempts": telemetry.get("attempts"),
                "candidates_considered": telemetry.get("candidates_considered"),
                "llm_invoked": telemetry.get("llm_invoked"),
                "deterministic_links": telemetry.get("deterministic_links"),
                "llm_links": telemetry.get("llm_links"),
                "room_group_count": len(room_groups),
            },
            "room_groups": room_groups,
        }


@router.get("/{run_id}/map-rooms", response_model=list[RoomLinkResponse])
def get_room_map(run_id: str, min_confidence: float = 0.6, include_groups: bool = False):
    with SessionLocal() as db:
        links = list(db.scalars(
            select(RoomMap)
            .where(RoomMap.run_id == run_id, RoomMap.confidence >= min_confidence)
            .order_by(RoomMap.confidence.desc())
        ))
        payload = [
            {
                "room_a": l.room_a,
                "room_b": l.room_b,
                "confidence": l.confidence,
                "rationale": l.rationale,
            }
            for l in links
        ]
        if not include_groups:
            return payload

        rooms_a = list(db.scalars(select(LineItem.room).where(LineItem.run_id == run_id, LineItem.doc == "A").distinct()))
        rooms_b = list(db.scalars(select(LineItem.room).where(LineItem.run_id == run_id, LineItem.doc == "B").distinct()))
        room_groups = build_room_groups_from_links(
            rooms_a=rooms_a,
            rooms_b=rooms_b,
            links=[
                RoomLink(room_a=l.room_a, room_b=l.room_b, confidence=float(l.confidence), rationale=l.rationale or "")
                for l in links
            ],
            min_confidence=max(0.40, min_confidence - 0.12),
        )
        return {"links": payload, "room_groups": room_groups}
