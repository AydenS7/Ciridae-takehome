from __future__ import annotations

from typing import Any, Iterable
from .llm_client import get_client
from .llm_match_schemas import MatchPlan

SYSTEM_PROMPT = """You are matching line items between two repair estimates within the SAME room/area.

Goal:
- For each contractor item (Doc A), either match it to at most ONE insurance item (Doc B), or mark it unmatched.
- Matching is based on scope similarity, not just price.
- Do NOT match across rooms; assume lists provided are already room-scoped.

Rules:
- One-to-one: each B item can be used at most once.
- Prefer precision over recall: if uncertain, leave unmatched.
- Ignore minor wording differences; consider synonyms and equivalent work descriptions.
- If B combines multiple A items into one, match the single best A item and leave the rest unmatched (we’ll handle merges later).
- If A combines multiple B items, still choose the best single B item.

Return pairs for every A item (including unmatched with item_b_id=null).
"""

def _to_brief_items(items: list[dict[str, Any]], max_len: int = 180) -> list[dict[str, Any]]:
    out = []
    for it in items:
        desc = (it["description"] or "").strip()
        if len(desc) > max_len:
            desc = desc[: max_len - 3] + "..."
        out.append(
            {
                "id": it["id"],
                "room": it["room"],
                "desc": desc,
                "amount": it["amount"],
            }
        )
    return out

def propose_matches_for_room(room_a: str, room_b: str, items_a: list[dict[str, Any]], items_b: list[dict[str, Any]]) -> MatchPlan:
    client = get_client()

    # Keep prompts bounded (LLM-only but practical)
    # If rooms are huge, we chunk A and keep B as context.
    user = {
        "room_a": room_a,
        "room_b": room_b,
        "items_a": _to_brief_items(items_a),
        "items_b": _to_brief_items(items_b),
        "instructions": "Return a MatchPlan with one ProposedPair for each items_a entry.",
    }

    parsed = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user)},
        ],
        response_format=MatchPlan,
    ).choices[0].message.parsed

    return parsed
