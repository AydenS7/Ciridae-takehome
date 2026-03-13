"""Room mapping engine combining deterministic scoring with LLM adjudication."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from statistics import mean
from typing import Iterable

from .llm_client import get_client
from .llm_roommap_schemas import RoomLink, RoomMapResult
from .llm_utils import expand_model_variants
from .settings import settings

SYSTEM_PROMPT = """You map rooms/areas between two repair estimates.

Rules:
- Only map rooms that refer to the same physical area.
- Handle renames and splits/merges:
  - One contractor room may correspond to multiple insurance rooms and vice versa.
- Use ONLY the provided room strings (exactly as given). Do not invent new room names.
- Treat name variants as potential matches when one name is a more specific form of another
  (for example added qualifiers/prefixes/suffixes), but do not merge clearly distinct numbered rooms.
- Prefer high precision: include a link only if it's likely correct.
- Provide a short rationale for each link.

CRITICAL — do NOT merge rooms that are physically distinct spaces:
- "Bedroom" and "Bedroom Closet" are DIFFERENT rooms. A closet is a separate space.
- Any room name that adds a room-type word (Closet, Bathroom, Kitchen, Garage, Laundry, Pantry, etc.)
  to another name describes a DIFFERENT room, not the same one.
- Only link rooms when the names describe the same physical space.
"""


_ROOMMAP_ALIAS_MAP: dict[str, str] = {
    "openai": "openai/gpt-4.1",
    "chatgpt": "openai/gpt-4.1",
    "gpt5.3": "openai/gpt-5.3",
    "gpt-5.3": "openai/gpt-5.3",
    "anthropic": "anthropic/claude-opus-4-5",
    "claude": "anthropic/claude-opus-4-5",
    "gemini": "gemini/gemini-2.5-pro",
    "google": "gemini/gemini-2.5-pro",
}

# Words that define a room TYPE — if one name has such a word and the other doesn't,
# they are clearly different physical spaces and must not be merged.
_ROOM_TYPE_DISTINGUISHERS: set[str] = {
    "closet",
    "bathroom",
    "bath",
    "kitchen",
    "garage",
    "basement",
    "laundry",
    "pantry",
    "foyer",
    "entry",
    "porch",
    "deck",
    "attic",
    "office",
    "linen",
}

_ROOM_STOPWORDS: set[str] = {
    "room",
    "area",
    "level",
    "floor",
    "story",
    "section",
    "zone",
    "main",
    "upper",
    "lower",
    "first",
    "second",
    "third",
    "the",
    "and",
    "of",
}

_DETERMINISTIC_ACCEPT_SCORE = 0.82
_DETERMINISTIC_AMBIGUOUS_SCORE = 0.58
_DETERMINISTIC_NAME_BONUS_SCORE = 0.78
_DETERMINISTIC_PROFILE_BONUS_SCORE = 0.15
_LLM_CANDIDATE_LIMIT_PER_A = 5
_MIN_LINK_CONFIDENCE_TO_KEEP = 0.42
_MAX_LINKS_PER_ROOM_A = 5
_MAX_LINKS_PER_ROOM_B = 5
_GROUP_CONFIDENCE_FLOOR = 0.48


class _RoomPair:
    __slots__ = (
        "room_a",
        "room_b",
        "name_similarity",
        "profile_similarity",
        "combined_score",
    )

    def __init__(self, room_a: str, room_b: str, name_similarity: float, profile_similarity: float, combined_score: float):
        self.room_a = room_a
        self.room_b = room_b
        self.name_similarity = name_similarity
        self.profile_similarity = profile_similarity
        self.combined_score = combined_score


def _candidate_model_variants(model_name: str) -> list[str]:
    return expand_model_variants(model_name or settings.roommap_model, _ROOMMAP_ALIAS_MAP)


def _room_name_tokens(name: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", (name or "").lower())
    return [tok for tok in raw if tok not in _ROOM_STOPWORDS]


def _room_number_tokens(name: str) -> set[str]:
    return set(re.findall(r"\d+", (name or "").lower()))


def _set_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _room_name_similarity(a: str, b: str) -> float:
    ta = _room_name_tokens(a)
    tb = _room_name_tokens(b)
    if not ta or not tb:
        return 0.0

    nums_a = _room_number_tokens(a)
    nums_b = _room_number_tokens(b)
    if nums_a and nums_b and nums_a != nums_b:
        return 0.0

    sa = set(ta)
    sb = set(tb)

    # If the symmetric difference contains a room-type distinguisher, these are physically
    # different spaces (e.g. "Bedroom" vs "Bedroom Closet"). Hard-cap the score so they
    # never reach the deterministic-accept or even ambiguous threshold.
    unique_to_a = sa - sb
    unique_to_b = sb - sa
    if (unique_to_a | unique_to_b) & _ROOM_TYPE_DISTINGUISHERS:
        return 0.25

    # Use Jaccard (inter/union) instead of containment ratio so that "Bedroom" vs
    # "Master Bedroom" scores ~0.5 rather than 1.0.
    jaccard = _set_jaccard(sa, sb)
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    suffix_match_bonus = 0.92 if ta and tb and ta[-1] == tb[-1] else 0.0
    return max(jaccard, seq, suffix_match_bonus)


def _normalize_profile_tokens(tokens: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        for piece in re.findall(r"[a-z0-9']+", str(token or "").lower()):
            if len(piece) <= 2:
                continue
            out.add(piece)
    return out


def _room_profile_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return _set_jaccard(tokens_a, tokens_b)


def _rank_room_pairs(
    rooms_a: list[str],
    rooms_b: list[str],
    room_profile_tokens_a: dict[str, set[str]],
    room_profile_tokens_b: dict[str, set[str]],
) -> list[_RoomPair]:
    ranked: list[_RoomPair] = []
    for room_a in rooms_a:
        for room_b in rooms_b:
            name_similarity = _room_name_similarity(room_a, room_b)
            profile_similarity = _room_profile_similarity(
                room_profile_tokens_a.get(room_a, set()),
                room_profile_tokens_b.get(room_b, set()),
            )
            combined = (0.78 * name_similarity) + (0.22 * profile_similarity)
            ranked.append(_RoomPair(room_a, room_b, name_similarity, profile_similarity, combined))
    ranked.sort(key=lambda pair: pair.combined_score, reverse=True)
    return ranked


def _deterministic_links_and_ambiguous_pairs(
    ranked_pairs: list[_RoomPair],
) -> tuple[list[RoomLink], list[dict[str, object]], dict[str, list[_RoomPair]]]:
    by_room_a: dict[str, list[_RoomPair]] = {}
    for pair in ranked_pairs:
        by_room_a.setdefault(pair.room_a, []).append(pair)

    deterministic_links: list[RoomLink] = []
    ambiguous_pairs: list[dict[str, object]] = []

    for room_a, pairs in by_room_a.items():
        accepted_for_a = 0
        for pair in pairs[: _LLM_CANDIDATE_LIMIT_PER_A + 3]:
            deterministic_accept = bool(
                pair.combined_score >= _DETERMINISTIC_ACCEPT_SCORE
                or (
                    pair.name_similarity >= _DETERMINISTIC_NAME_BONUS_SCORE
                    and pair.profile_similarity >= _DETERMINISTIC_PROFILE_BONUS_SCORE
                )
            )
            if deterministic_accept and accepted_for_a < 3:
                conf = max(0.60, min(0.97, (0.68 * pair.combined_score) + (0.24 * pair.name_similarity) + 0.08))
                deterministic_links.append(
                    RoomLink(
                        room_a=pair.room_a,
                        room_b=pair.room_b,
                        confidence=conf,
                        rationale=(
                            "deterministic-room-link: strong room-name/profile alignment "
                            f"(name={pair.name_similarity:.2f}, profile={pair.profile_similarity:.2f})"
                        ),
                    )
                )
                accepted_for_a += 1
                continue

            ambiguous = bool(
                pair.combined_score >= _DETERMINISTIC_AMBIGUOUS_SCORE
                or pair.name_similarity >= max(0.66, _DETERMINISTIC_AMBIGUOUS_SCORE + 0.08)
                or pair.profile_similarity >= 0.26
            )
            if ambiguous:
                ambiguous_pairs.append(
                    {
                        "room_a": pair.room_a,
                        "room_b": pair.room_b,
                        "name_similarity": round(pair.name_similarity, 4),
                        "profile_similarity": round(pair.profile_similarity, 4),
                        "combined_score": round(pair.combined_score, 4),
                    }
                )

    return deterministic_links, ambiguous_pairs, by_room_a


def _merge_links(*link_groups: Iterable[RoomLink]) -> list[RoomLink]:
    by_pair: dict[tuple[str, str], RoomLink] = {}
    for group in link_groups:
        for link in group:
            key = (link.room_a, link.room_b)
            existing = by_pair.get(key)
            if existing is None:
                by_pair[key] = link
                continue
            if float(link.confidence) > float(existing.confidence):
                if existing.rationale and link.rationale and existing.rationale not in link.rationale:
                    merged_reason = f"{link.rationale}; {existing.rationale}"
                else:
                    merged_reason = link.rationale or existing.rationale
                by_pair[key] = RoomLink(
                    room_a=link.room_a,
                    room_b=link.room_b,
                    confidence=float(link.confidence),
                    rationale=merged_reason,
                )
            elif existing.rationale and link.rationale and link.rationale not in existing.rationale:
                by_pair[key] = RoomLink(
                    room_a=existing.room_a,
                    room_b=existing.room_b,
                    confidence=float(existing.confidence),
                    rationale=f"{existing.rationale}; {link.rationale}",
                )

    merged = [lnk for lnk in by_pair.values() if float(lnk.confidence) >= _MIN_LINK_CONFIDENCE_TO_KEEP]
    merged.sort(key=lambda l: float(l.confidence), reverse=True)
    return merged


def _limit_links_per_room(links: list[RoomLink]) -> list[RoomLink]:
    kept: list[RoomLink] = []
    count_a: dict[str, int] = {}
    count_b: dict[str, int] = {}
    for link in sorted(links, key=lambda l: float(l.confidence), reverse=True):
        a_used = count_a.get(link.room_a, 0)
        b_used = count_b.get(link.room_b, 0)
        if a_used >= _MAX_LINKS_PER_ROOM_A and b_used >= _MAX_LINKS_PER_ROOM_B:
            continue
        kept.append(link)
        count_a[link.room_a] = a_used + 1
        count_b[link.room_b] = b_used + 1
    return kept


def _connected_room_groups(rooms_a: list[str], rooms_b: list[str], links: list[RoomLink], *, min_confidence: float = _GROUP_CONFIDENCE_FLOOR) -> list[dict[str, object]]:
    graph: dict[str, set[str]] = {}

    def add_edge(left: str, right: str) -> None:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)

    for room in rooms_a:
        graph.setdefault(f"A::{room}", set())
    for room in rooms_b:
        graph.setdefault(f"B::{room}", set())

    link_by_pair: dict[tuple[str, str], float] = {}
    for link in links:
        conf = float(link.confidence)
        link_by_pair[(link.room_a, link.room_b)] = max(conf, link_by_pair.get((link.room_a, link.room_b), 0.0))
        if conf >= min_confidence:
            add_edge(f"A::{link.room_a}", f"B::{link.room_b}")

    groups: list[dict[str, object]] = []
    visited: set[str] = set()
    idx = 0

    for node in graph:
        if node in visited:
            continue
        idx += 1
        stack = [node]
        component_nodes: list[str] = []
        component_edges: list[float] = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            component_nodes.append(cur)
            for nxt in graph.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

        rooms_group_a = sorted({n[3:] for n in component_nodes if n.startswith("A::")})
        rooms_group_b = sorted({n[3:] for n in component_nodes if n.startswith("B::")})

        for room_a in rooms_group_a:
            for room_b in rooms_group_b:
                conf = link_by_pair.get((room_a, room_b))
                if conf is not None:
                    component_edges.append(conf)

        groups.append(
            {
                "group_id": idx,
                "rooms_a": rooms_group_a,
                "rooms_b": rooms_group_b,
                "link_count": len(component_edges),
                "max_confidence": max(component_edges) if component_edges else 0.0,
                "avg_confidence": mean(component_edges) if component_edges else 0.0,
            }
        )

    groups.sort(key=lambda g: (len(g["rooms_a"]) + len(g["rooms_b"]), float(g["max_confidence"])), reverse=True)
    return groups


def _map_rooms_via_llm(
    rooms_a: list[str],
    rooms_b: list[str],
    *,
    ambiguous_pairs: list[dict[str, object]] | None = None,
) -> tuple[RoomMapResult, dict]:
    client = get_client()

    user_payload = {
        "rooms_a": rooms_a,
        "rooms_b": rooms_b,
        "instructions": "Return links that connect equivalent physical areas. If unsure, omit.",
    }
    if ambiguous_pairs:
        user_payload["candidate_pairs"] = ambiguous_pairs
        user_payload["instructions"] = (
            "Decide ONLY from candidate_pairs. Keep only plausible equivalents; omit uncertain links. "
            "Do not introduce pairs outside candidate_pairs."
        )

    last_error: Exception | None = None
    model_candidates: list[str] = []
    seen: set[str] = set()
    for name in (
        settings.roommap_model,
        settings.matching_first_pass_model,
        "openai/gpt-5",
        "openai/gpt-5.1",
        "anthropic/claude-opus-4-5",
        "gemini/gemini-2.5-pro",
        "openai/gpt-4.1",
        "openai/gpt-4o-2024-08-06",
    ):
        for candidate in _candidate_model_variants(name):
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            model_candidates.append(candidate)

    attempts = 0
    for model_candidate in model_candidates:
        attempts += 1
        try:
            result = client.chat.completions.parse(
                model=model_candidate,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
                ],
                response_format=RoomMapResult,
            ).choices[0].message.parsed
            telemetry = {
                "model_used": model_candidate,
                "attempts": attempts,
                "candidates_considered": len(model_candidates),
            }
            return result, telemetry
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("No model candidate available for room mapping.")


def map_rooms_via_llm(
    rooms_a: list[str],
    rooms_b: list[str],
    *,
    room_profile_tokens_a: dict[str, set[str]] | None = None,
    room_profile_tokens_b: dict[str, set[str]] | None = None,
) -> tuple[RoomMapResult, dict]:
    """
    Hybrid room mapping:
    1) Deterministic scorer produces high-confidence links + ambiguous candidates.
    2) LLM adjudicates only ambiguous candidates.
    3) Return merged top links and computed room groups.
    """
    room_profile_tokens_a = {
        room: _normalize_profile_tokens(tokens)
        for room, tokens in (room_profile_tokens_a or {}).items()
    }
    room_profile_tokens_b = {
        room: _normalize_profile_tokens(tokens)
        for room, tokens in (room_profile_tokens_b or {}).items()
    }

    for room in rooms_a:
        room_profile_tokens_a.setdefault(room, set())
    for room in rooms_b:
        room_profile_tokens_b.setdefault(room, set())

    # Guarantee: rooms with identical normalized names always map (e.g. "Main Level" → "Main Level").
    # This runs before deterministic scoring, which can miss exact matches when all name tokens
    # are stopwords (e.g. "main" and "level" are both stopwords → name_similarity = 0.0).
    exact_match_links: list[RoomLink] = []
    for room_a in rooms_a:
        a_norm = room_a.strip().lower()
        for room_b in rooms_b:
            if room_b.strip().lower() == a_norm:
                exact_match_links.append(
                    RoomLink(
                        room_a=room_a,
                        room_b=room_b,
                        confidence=0.99,
                        rationale="exact-name-match",
                    )
                )

    ranked_pairs = _rank_room_pairs(
        rooms_a,
        rooms_b,
        room_profile_tokens_a,
        room_profile_tokens_b,
    )
    deterministic_links, ambiguous_pairs, _ = _deterministic_links_and_ambiguous_pairs(ranked_pairs)

    llm_links: list[RoomLink] = []
    llm_telemetry: dict[str, object] = {
        "model_used": "none",
        "attempts": 0,
        "candidates_considered": 0,
        "invoked": False,
    }

    if ambiguous_pairs:
        # Keep candidate set bounded so room mapping remains fast.
        limited_ambiguous: list[dict[str, object]] = []
        per_a_count: dict[str, int] = {}
        for pair in sorted(ambiguous_pairs, key=lambda p: float(p.get("combined_score", 0.0)), reverse=True):
            room_a = str(pair["room_a"])
            used = per_a_count.get(room_a, 0)
            if used >= _LLM_CANDIDATE_LIMIT_PER_A:
                continue
            limited_ambiguous.append(pair)
            per_a_count[room_a] = used + 1

        if limited_ambiguous:
            llm_result, llm_telemetry = _map_rooms_via_llm(
                rooms_a,
                rooms_b,
                ambiguous_pairs=limited_ambiguous,
            )
            llm_links = list(llm_result.links)
            llm_telemetry["invoked"] = True
            llm_telemetry["ambiguous_pairs_sent"] = len(limited_ambiguous)

    merged_links = _limit_links_per_room(_merge_links(exact_match_links, deterministic_links, llm_links))
    result = RoomMapResult(links=merged_links)
    groups = _connected_room_groups(rooms_a, rooms_b, merged_links)

    telemetry: dict[str, object] = {
        "model_used": llm_telemetry.get("model_used", "none"),
        "attempts": int(llm_telemetry.get("attempts", 0)),
        "candidates_considered": int(llm_telemetry.get("candidates_considered", 0)),
        "llm_invoked": bool(llm_telemetry.get("invoked", False)),
        "deterministic_links": len(deterministic_links),
        "llm_links": len(llm_links),
        "final_links": len(merged_links),
        "room_groups": groups,
    }
    return result, telemetry


def build_room_groups_from_links(
    rooms_a: list[str],
    rooms_b: list[str],
    links: Iterable[RoomLink],
    *,
    min_confidence: float = _GROUP_CONFIDENCE_FLOOR,
) -> list[dict[str, object]]:
    return _connected_room_groups(rooms_a, rooms_b, list(links), min_confidence=min_confidence)
