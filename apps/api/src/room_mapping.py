from __future__ import annotations
from typing import Iterable
from .llm_client import get_client
from .llm_roommap_schemas import RoomMapResult

SYSTEM_PROMPT = """You map rooms/areas between two repair estimates.

Rules:
- Only map rooms that refer to the same physical area.
- Handle renames and splits/merges:
  - One contractor room may correspond to multiple insurance rooms and vice versa.
- Use ONLY the provided room strings (exactly as given). Do not invent new room names.
- Prefer high precision: include a link only if it's likely correct.
- Provide a short rationale for each link.
"""

def map_rooms_via_llm(rooms_a: list[str], rooms_b: list[str]) -> RoomMapResult:
    client = get_client()

    user = f"""Contractor rooms (Doc A):
{rooms_a}

Insurance rooms (Doc B):
{rooms_b}

Return links that connect equivalent physical areas. If unsure, omit.
"""

    parsed = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format=RoomMapResult,
    ).choices[0].message.parsed

    return parsed
