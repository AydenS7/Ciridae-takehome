"""Endpoints and helper logic for room-scoped line-item matching classification."""

from __future__ import annotations

import heapq
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from time import perf_counter

from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, func

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .models_roommap import RoomMap
from .models_matches import Match
from .matching_llm import (
    get_match_telemetry,
    propose_matches_for_room,
    propose_matches_for_room_ensemble,
)
from .settings import settings

router = APIRouter(prefix="/runs", tags=["matching"])

def _pct_diff(a: float, b: float) -> float:
    # avoid div by zero; treat as infinite diff if either missing or zero-ish
    if a is None or b is None:
        return float("inf")
    denom = max(abs(b), 1e-9)
    return abs(a - b) / denom


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "per",
    "to",
    "of",
    "in",
    "on",
    "by",
    "at",
    "or",
    "a",
    "an",
}

_CRITICAL_BLUE_KEYWORDS = {
    "megohmmeter",
    "electrical",
    "testing",
    "test",
    "permit",
    "code",
    "mold",
    "asbestos",
    "lead",
    "engineer",
}

_UNIT_CANONICAL: dict[str, str] = {
    "EA": "EA",
    "EACH": "EA",
    "HR": "HR",
    "HOUR": "HR",
    "HOURS": "HR",
    "HRS": "HR",
    "SF": "SF",
    "SQFT": "SF",
    "SQUAREFEET": "SF",
    "LF": "LF",
    "LINFT": "LF",
    "LINEARFEET": "LF",
    "SY": "SY",
    "SQYD": "SY",
    "SQUAREYARDS": "SY",
}


def _tokens(text: str) -> set[str]:
    toks = {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2}
    return {t for t in toks if t not in _STOPWORDS}


def _canonical_unit(unit: str | None) -> str:
    raw = (unit or "").strip().upper()
    if not raw:
        return ""
    key = re.sub(r"[^A-Z0-9]", "", raw)
    return _UNIT_CANONICAL.get(key, key)


def _has_meaningful_description(text: str) -> bool:
    toks = _tokens(text or "")
    if len(toks) < 2:
        return False
    alpha = sum(1 for ch in (text or "") if ch.isalpha())
    return alpha >= 4


def _is_critical_blue_description(text: str) -> bool:
    toks = _tokens(text or "")
    if not toks:
        return False
    return any(k in toks for k in _CRITICAL_BLUE_KEYWORDS)


def _token_jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union


def _amount_closeness(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    d = _pct_diff(a, b)
    if d <= 0.02:
        return 1.0
    if d <= 0.1:
        return 0.75
    if d <= 0.25:
        return 0.45
    if d <= 0.5:
        return 0.2
    return 0.0


def _token_overlap_on_shorter(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(1, min(len(ta), len(tb)))


def _set_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _normalized_name(text: str) -> str:
    toks = [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2 and t not in _STOPWORDS]
    return " ".join(toks)


def _name_similarity_score(a_desc: str, b_desc: str) -> float:
    desc_sim = _token_jaccard(a_desc, b_desc)
    desc_overlap = _token_overlap_on_shorter(a_desc, b_desc)
    norm_a = _normalized_name(a_desc)
    norm_b = _normalized_name(b_desc)
    seq_sim = SequenceMatcher(None, norm_a, norm_b).ratio() if norm_a and norm_b else 0.0
    return max(desc_sim, desc_overlap, seq_sim)


def _room_name_tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (name or "").lower())


def _room_number_tokens(name: str) -> set[str]:
    return set(re.findall(r"\d+", (name or "").lower()))


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
    overlap_shorter = len(sa & sb) / max(1, min(len(sa), len(sb)))
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    tail_bonus = 0.92 if ta[-1] == tb[-1] else 0.0
    return max(overlap_shorter, seq, tail_bonus)


def _expand_candidate_rooms(
    seed_rooms: list[str],
    all_rooms: list[str],
    room_desc_tokens: dict[str, set[str]],
) -> list[str]:
    out = list(dict.fromkeys(seed_rooms))
    seen = set(out)

    for base in list(out):
        for candidate in all_rooms:
            if candidate in seen:
                continue
            name_sim = _room_name_similarity(base, candidate)
            if name_sim < 0.74:
                continue
            profile_sim = _set_jaccard(
                room_desc_tokens.get(base, set()),
                room_desc_tokens.get(candidate, set()),
            )
            if name_sim >= 0.90 or (name_sim >= 0.80 and profile_sim >= 0.06):
                out.append(candidate)
                seen.add(candidate)
    return out


def _build_room_group_candidates(
    room_links: list[RoomMap],
    rooms_a: list[str],
    rooms_b: list[str],
    *,
    min_confidence: float,
) -> tuple[dict[str, list[str]], int]:
    graph: dict[str, set[str]] = {}

    def _add_edge(a_node: str, b_node: str) -> None:
        graph.setdefault(a_node, set()).add(b_node)
        graph.setdefault(b_node, set()).add(a_node)

    for room in rooms_a:
        graph.setdefault(f"A::{room}", set())
    for room in rooms_b:
        graph.setdefault(f"B::{room}", set())

    for link in room_links:
        conf = float(link.confidence or 0.0)
        if conf < min_confidence:
            continue
        _add_edge(f"A::{link.room_a}", f"B::{link.room_b}")

    room_b_by_room_a: dict[str, set[str]] = {room: set() for room in rooms_a}
    visited: set[str] = set()
    component_count = 0

    for node in graph:
        if node in visited:
            continue
        component_count += 1
        stack = [node]
        visited.add(node)
        nodes: list[str] = []
        while stack:
            cur = stack.pop()
            nodes.append(cur)
            for nxt in graph.get(cur, set()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)

        group_a = [n[3:] for n in nodes if n.startswith("A::")]
        group_b = [n[3:] for n in nodes if n.startswith("B::")]
        if not group_a or not group_b:
            continue
        for room_a in group_a:
            room_b_by_room_a.setdefault(room_a, set()).update(group_b)

    materialized = {
        room_a: sorted(list(room_bs))
        for room_a, room_bs in room_b_by_room_a.items()
        if room_bs
    }
    return materialized, component_count


def _soft_room_candidates(
    room_a: str,
    all_rooms_b: list[str],
    a_room_desc_tokens: dict[str, set[str]],
    b_room_desc_tokens: dict[str, set[str]],
    *,
    limit: int = 6,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    a_tokens = a_room_desc_tokens.get(room_a, set())
    for room_b in all_rooms_b:
        name_sim = _room_name_similarity(room_a, room_b)
        profile_sim = _set_jaccard(a_tokens, b_room_desc_tokens.get(room_b, set()))
        score = (0.72 * name_sim) + (0.28 * profile_sim)
        if score < 0.20 and name_sim < 0.34 and profile_sim < 0.10:
            continue
        ranked.append((score, room_b))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top = [room for _, room in ranked[: max(1, limit)]]
    if top:
        return top
    return list(all_rooms_b[: max(1, min(limit, len(all_rooms_b)))])


def _pair_similarity(a_item: LineItem, b_item: LineItem) -> tuple[float, float, float]:
    desc_sim = _token_jaccard(a_item.description or "", b_item.description or "")
    amt_sim = _amount_closeness(a_item.amount, b_item.amount)
    score = (0.78 * desc_sim) + (0.22 * amt_sim)
    return score, desc_sim, amt_sim


def _semantic_same(
    a_desc: str,
    b_desc: str,
    *,
    scope_same_hint: bool,
    scope_sim_threshold: float,
    scope_overlap_threshold: float,
) -> tuple[bool, float, float]:
    desc_sim = _token_jaccard(a_desc, b_desc)
    desc_overlap = _token_overlap_on_shorter(a_desc, b_desc)
    seq_sim = _name_similarity_score(a_desc, b_desc)
    semantic_from_text = bool(
        desc_sim >= scope_sim_threshold
        or desc_overlap >= scope_overlap_threshold
        or seq_sim >= max(0.58, scope_sim_threshold + 0.10)
    )
    # Include LLM scope hint to recover paraphrase cases that are semantically same but lexically far apart.
    # Keep a tiny lexical floor to avoid obvious random pairings.
    semantic_from_hint = bool(scope_same_hint) and bool(max(desc_sim, desc_overlap, seq_sim) >= 0.08)
    semantic = bool(semantic_from_text or semantic_from_hint)
    return semantic, desc_sim, desc_overlap


@dataclass
class _MCFEdge:
    to: int
    rev: int
    cap: int
    cost: int


def _add_mcf_edge(graph: list[list[_MCFEdge]], u: int, v: int, cap: int, cost: int) -> None:
    graph[u].append(_MCFEdge(to=v, rev=len(graph[v]), cap=cap, cost=cost))
    graph[v].append(_MCFEdge(to=u, rev=len(graph[u]) - 1, cap=0, cost=-cost))


def _min_cost_max_flow(
    graph: list[list[_MCFEdge]],
    source: int,
    sink: int,
    max_flow: int,
) -> tuple[int, int]:
    n = len(graph)
    flow = 0
    total_cost = 0

    while flow < max_flow:
        dist = [10**18] * n
        parent_v = [-1] * n
        parent_e = [-1] * n
        dist[source] = 0
        heap: list[tuple[int, int]] = [(0, source)]

        while heap:
            d, v = heapq.heappop(heap)
            if d != dist[v]:
                continue
            for ei, e in enumerate(graph[v]):
                if e.cap <= 0:
                    continue
                nd = d + e.cost
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    parent_v[e.to] = v
                    parent_e[e.to] = ei
                    heapq.heappush(heap, (nd, e.to))

        if dist[sink] >= 10**18:
            break

        add_flow = max_flow - flow
        v = sink
        while v != source:
            pv = parent_v[v]
            pe = parent_e[v]
            if pv < 0 or pe < 0:
                add_flow = 0
                break
            add_flow = min(add_flow, graph[pv][pe].cap)
            v = pv

        if add_flow <= 0:
            break

        v = sink
        while v != source:
            pv = parent_v[v]
            pe = parent_e[v]
            e = graph[pv][pe]
            e.cap -= add_flow
            rev = graph[v][e.rev]
            rev.cap += add_flow
            v = pv

        flow += add_flow
        total_cost += add_flow * dist[sink]

    return flow, total_cost


def _optimize_room_assignment(
    *,
    a_ids: list[int],
    b_ids: list[int],
    score_by_a: dict[int, dict[int, float]],
    unmatched_threshold: float,
) -> dict[int, int | None]:
    assignment: dict[int, int | None] = {a_id: None for a_id in a_ids}
    if not a_ids:
        return assignment
    if not b_ids:
        return assignment

    na = len(a_ids)
    nb = len(b_ids)
    source = 0
    a0 = 1
    b0 = a0 + na
    sink = b0 + nb
    graph: list[list[_MCFEdge]] = [[] for _ in range(sink + 1)]

    a_index = {a_id: a0 + i for i, a_id in enumerate(a_ids)}
    b_index = {b_id: b0 + i for i, b_id in enumerate(b_ids)}
    b_index_rev = {idx: b_id for b_id, idx in b_index.items()}
    unmatched_cost = int(round((1.0 - max(0.0, min(1.0, unmatched_threshold))) * 1000))

    for a_id in a_ids:
        ai = a_index[a_id]
        _add_mcf_edge(graph, source, ai, 1, 0)
        _add_mcf_edge(graph, ai, sink, 1, unmatched_cost)
        for b_id, score in score_by_a.get(a_id, {}).items():
            if b_id not in b_index:
                continue
            # Lower cost = better. Map score directly to [0,1] so score > unmatched_threshold wins.
            clamped = max(0.0, min(1.0, float(score)))
            edge_cost = int(round((1.0 - clamped) * 1000))
            _add_mcf_edge(graph, ai, b_index[b_id], 1, edge_cost)

    for b_id in b_ids:
        _add_mcf_edge(graph, b_index[b_id], sink, 1, 0)

    _min_cost_max_flow(graph, source, sink, len(a_ids))

    for a_id in a_ids:
        ai = a_index[a_id]
        chosen_b: int | None = None
        for e in graph[ai]:
            if e.to < b0 or e.to >= sink:
                continue
            rev = graph[e.to][e.rev]
            if rev.cap > 0:
                chosen_b = b_index_rev[e.to]
                break
        assignment[a_id] = chosen_b

    return assignment


def _choice_quality(
    *,
    a_item: LineItem,
    choice: tuple[int | None, bool, float, str] | None,
    b_by_id: dict[int, LineItem],
    green_amount_tol: float,
    scope_sim_threshold: float,
    scope_overlap_threshold: float,
) -> float:
    if choice is None:
        return float("-inf")
    b_id, _scope_same, conf, _ = choice
    if b_id is None or b_id not in b_by_id:
        return float("-inf")

    b_item = b_by_id[b_id]
    desc_sim = _token_jaccard(a_item.description or "", b_item.description or "")
    desc_overlap = _token_overlap_on_shorter(a_item.description or "", b_item.description or "")
    amt_sim = _amount_closeness(a_item.amount, b_item.amount)
    a_amt = a_item.amount
    b_amt = b_item.amount
    within_tol = (_pct_diff(a_amt, b_amt) <= green_amount_tol) if (a_amt is not None and b_amt is not None) else False
    semantic = bool(desc_sim >= scope_sim_threshold or desc_overlap >= scope_overlap_threshold)

    score = (0.45 * float(conf)) + (0.35 * desc_sim) + (0.20 * amt_sim)
    if semantic:
        score += 0.14
    if within_tol:
        score += 0.26
    return score


def _any_number_match(a_item: LineItem, b_item: LineItem, tol: float) -> bool:
    """
    Count how many numeric values from A cross-match numeric values from B within tolerance.
    Each A value and B value can only be used once (greedy best-first).
    Fields compared: amount, unit_price, quantity — any combination counts.
    Returns the number of matched pairs.
    """
    a_vals = [v for v in (a_item.amount, a_item.unit_price, a_item.quantity) if v is not None and v > 0]
    b_vals = [v for v in (b_item.amount, b_item.unit_price, b_item.quantity) if v is not None and v > 0]
    used_b: set[int] = set()
    matches = 0
    # Try each A value against all unused B values; take first match found.
    for av in a_vals:
        for bi, bv in enumerate(b_vals):
            if bi not in used_b and _pct_diff(av, bv) <= tol:
                used_b.add(bi)
                matches += 1
                break
    return matches


def _metadata_tolerance_result(
    a_item: LineItem,
    b_item: LineItem,
    *,
    tol: float,
) -> tuple[bool, bool, list[str]]:
    """
    Returns:
    - metadata_within_tol: 3+ numeric cross-matches found (green threshold)
    - amount_within_tol: same as metadata_within_tol (kept for compatibility)
    - mismatch_fields: informational — fields present on both sides that differ beyond tol
    """
    match_count = _any_number_match(a_item, b_item, tol)
    green = match_count >= 3

    # Informational mismatch list for rationale string only.
    mismatch_fields: list[str] = []
    for name, av, bv in [
        ("amount", a_item.amount, b_item.amount),
        ("quantity", a_item.quantity, b_item.quantity),
        ("unit_price", a_item.unit_price, b_item.unit_price),
    ]:
        if av is not None and bv is not None and _pct_diff(av, bv) > tol:
            mismatch_fields.append(name)

    return green, green, mismatch_fields


def _nugget_signature(room_b: str, description: str, amount: float | None) -> tuple[str, str, float | None]:
    desc = (description or "").lower()
    desc = re.sub(r"^\s*\d+\.\s*", "", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    amt = round(float(amount), 2) if amount is not None else None
    return ((room_b or "").strip().lower(), desc, amt)


def _fallback_match(
    a_item: LineItem,
    b_items: list[LineItem],
    used_b: set[int],
    *,
    min_score: float = 0.42,
) -> tuple[int | None, float, str]:
    best_id: int | None = None
    best_score = 0.0
    best_reason = ""

    a_desc = a_item.description or ""
    for b in b_items:
        if b.id in used_b:
            continue
        desc_sim = _token_jaccard(a_desc, b.description or "")
        amt_sim = _amount_closeness(a_item.amount, b.amount)
        score = (0.78 * desc_sim) + (0.22 * amt_sim)

        if score > best_score:
            best_score = score
            best_id = b.id
            best_reason = f"fallback similarity={score:.2f} (desc={desc_sim:.2f}, amount={amt_sim:.2f})"

    # Conservative threshold to avoid false matches.
    if best_id is None or best_score < min_score:
        return None, 0.0, ""
    return best_id, best_score, best_reason


def _best_pairs_by_a(plan) -> dict[int, tuple[int | None, bool, float, str]]:
    best: dict[int, tuple[int | None, bool, float, str]] = {}
    for p in plan.pairs:
        cur = best.get(p.item_a_id)
        candidate = (p.item_b_id, bool(p.scope_same), float(p.confidence), p.rationale or "")
        if cur is None or candidate[2] > cur[2]:
            best[p.item_a_id] = candidate
    return best


def _price_proximity_rescue(
    a_items: list,
    b_by_id: dict,
    assigned_for_a: dict,
    globally_used_b_ids: set,
    *,
    green_price_tol: float = 0.05,
    orange_price_tol: float = 0.15,
    min_desc_sim_green: float = 0.40,
    min_desc_sim_orange: float = 0.20,
) -> None:
    """
    Final rescue pass for still-unmatched A items: search unused B items by price proximity.
    If price is close AND descriptions have some overlap, force orange or green.
    Mutates assigned_for_a and globally_used_b_ids in-place.
    """
    unmatched_a = [a for a in a_items if assigned_for_a.get(a.id, {}).get("item_b_id") is None]
    if not unmatched_a:
        return
    available_b = [b for b in b_by_id.values() if b.id not in globally_used_b_ids]
    if not available_b:
        return

    for a_item in unmatched_a:
        a_amt = a_item.amount
        if a_amt is None:
            continue

        best_b = None
        best_composite = 0.0
        best_price_diff = float("inf")
        best_forced_status: str | None = None

        for b_item in available_b:
            b_amt = b_item.amount
            if b_amt is None:
                continue
            price_diff = _pct_diff(a_amt, b_amt)
            if price_diff > orange_price_tol:
                continue

            desc_sim = _token_jaccard(a_item.description or "", b_item.description or "")
            amt_closeness = _amount_closeness(a_amt, b_amt)
            composite = (0.45 * desc_sim) + (0.55 * amt_closeness)

            if price_diff <= green_price_tol and desc_sim >= min_desc_sim_green:
                forced_status = "green"
            elif price_diff <= orange_price_tol and desc_sim >= min_desc_sim_orange:
                forced_status = "orange"
            else:
                continue

            if composite > best_composite or (
                composite == best_composite and price_diff < best_price_diff
            ):
                best_composite = composite
                best_price_diff = price_diff
                best_b = b_item
                best_forced_status = forced_status

        if best_b is not None and best_forced_status is not None:
            globally_used_b_ids.add(best_b.id)
            available_b = [b for b in available_b if b.id != best_b.id]
            desc_sim_final = _token_jaccard(a_item.description or "", best_b.description or "")
            current = assigned_for_a.get(a_item.id, {})
            rationales = list(current.get("rationales", []))
            rationales.append(
                f"price-proximity rescue → {best_forced_status} "
                f"(price_diff={best_price_diff:.1%}, desc_sim={desc_sim_final:.2f})"
            )
            assigned_for_a[a_item.id] = {
                "item_b_id": best_b.id,
                "scope_same": best_forced_status == "green",
                "confidence": max(float(current.get("confidence") or 0.0), best_composite),
                "rationales": rationales,
                "force_status": best_forced_status,
            }


def _critical_blue_by_a(plan) -> dict[int, bool]:
    """Extract the LLM's critical_blue assessment for unmatched A items."""
    result: dict[int, bool] = {}
    for p in plan.pairs:
        if p.item_b_id is None and getattr(p, "critical_blue", False):
            result[p.item_a_id] = True
    return result

def _run_llm_for_room(
    *,
    room_a: str,
    room_b: str,
    a_dicts: list,
    b_dicts: list,
    first_pass_unsure_conf: float,
    first_pass_unsure_null_conf: float,
    first_pass_scope_unsure_conf: float,
    force_review_on_null: bool,
) -> dict:
    """Run first- and second-pass LLM matching for one room. Called in parallel across rooms."""
    first_pass = propose_matches_for_room(room_a, room_b, a_dicts, b_dicts)
    first_best_for_a = _best_pairs_by_a(first_pass)
    first_critical_blue = _critical_blue_by_a(first_pass)

    first_pass_unsure_ids: set[int] = set()
    for a in a_dicts:
        a_id = int(a["id"])
        choice = first_best_for_a.get(a_id)
        if choice is None:
            first_pass_unsure_ids.add(a_id)
            continue
        item_b_id, scope_same, conf, _ = choice
        if item_b_id is None and force_review_on_null:
            first_pass_unsure_ids.add(a_id)
            continue
        if conf < first_pass_unsure_conf:
            first_pass_unsure_ids.add(a_id)
            continue
        if item_b_id is None and conf < first_pass_unsure_null_conf:
            first_pass_unsure_ids.add(a_id)
            continue
        if (not bool(scope_same)) and conf < first_pass_scope_unsure_conf:
            first_pass_unsure_ids.add(a_id)

    second_best_for_a: dict = {}
    second_critical_blue: dict = {}
    second_pass_reviewed_count = 0
    second_pass_rooms_invoked = 0

    if first_pass_unsure_ids:
        uncertain_a_dicts = [x for x in a_dicts if int(x["id"]) in first_pass_unsure_ids]
        first_pass_uncertain_by_a = {
            a_id: first_best_for_a[a_id]
            for a_id in first_pass_unsure_ids
            if a_id in first_best_for_a
        }
        second_pass = propose_matches_for_room_ensemble(
            room_a,
            room_b,
            uncertain_a_dicts,
            b_dicts,
            first_pass_by_a=first_pass_uncertain_by_a,
        )
        second_best_for_a = _best_pairs_by_a(second_pass)
        second_critical_blue = _critical_blue_by_a(second_pass)
        second_pass_reviewed_count = len(first_pass_unsure_ids)
        second_pass_rooms_invoked = 1

    return {
        "first_best_for_a": first_best_for_a,
        "first_critical_blue": first_critical_blue,
        "second_best_for_a": second_best_for_a,
        "second_critical_blue": second_critical_blue,
        "first_pass_total_evaluated": len(a_dicts),
        "first_pass_uncertain_count": len(first_pass_unsure_ids),
        "second_pass_reviewed_count": second_pass_reviewed_count,
        "second_pass_rooms_invoked": second_pass_rooms_invoked,
    }


@router.post("/{run_id}/match")
def match_run(run_id: str, min_room_confidence: float = 0.6):
    """
    LLM-only pairing per mapped room, deterministic classification:
    - comparisons use semantic name/description + key metadata (amount/quantity/unit/unit_price)
    - blue: A-only (no B match)
    - green: semantic scope match AND key metadata within ±2%
    - orange: semantic scope match but one or more key metadata fields differ beyond ±2%
    - nugget: insurance-only (B present with no matched A)
    """
    t0 = perf_counter()
    with SessionLocal() as db:
        first_pass_unsure_conf = float(settings.matching_first_pass_unsure_confidence)
        first_pass_unsure_null_conf = float(settings.matching_first_pass_unsure_null_confidence)
        first_pass_scope_unsure_conf = float(settings.matching_first_pass_scope_unsure_confidence)
        force_review_on_null = bool(settings.matching_force_review_on_null)
        green_amount_tol = float(settings.matching_green_amount_tolerance_pct)

        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        # must have extracted items
        n_items = db.query(func.count(LineItem.id)).filter(LineItem.run_id == run_id).scalar() or 0
        if n_items == 0:
            raise HTTPException(status_code=400, detail="no extracted items; run /extract first")

        # must have room mapping candidates
        room_links_all = (
            db.query(RoomMap)
            .filter(RoomMap.run_id == run_id)
            .order_by(RoomMap.confidence.desc())
            .all()
        )
        if not room_links_all:
            raise HTTPException(status_code=400, detail="no room mapping; run /map-rooms first")

        links_by_room_a: dict[str, list[RoomMap]] = {}
        best_room_a_for_room_b: dict[str, str] = {}
        best_room_a_score_for_room_b: dict[str, float] = {}
        for link in room_links_all:
            links_by_room_a.setdefault(link.room_a, []).append(link)
            conf = float(link.confidence or 0.0)
            prev_best = best_room_a_score_for_room_b.get(link.room_b, -1.0)
            if conf > prev_best:
                best_room_a_for_room_b[link.room_b] = link.room_a
                best_room_a_score_for_room_b[link.room_b] = conf

        rooms_a = [
            r[0]
            for r in db.query(LineItem.room)
            .filter(LineItem.run_id == run_id, LineItem.doc == "A")
            .distinct()
            .all()
        ]
        all_rooms_b = [
            r[0]
            for r in db.query(LineItem.room)
            .filter(LineItem.run_id == run_id, LineItem.doc == "B")
            .distinct()
            .all()
        ]
        a_room_desc_tokens: dict[str, set[str]] = {}
        for room_name, desc in (
            db.query(LineItem.room, LineItem.description)
            .filter(LineItem.run_id == run_id, LineItem.doc == "A")
            .all()
        ):
            if room_name not in a_room_desc_tokens:
                a_room_desc_tokens[room_name] = set()
            a_room_desc_tokens[room_name].update(_tokens(desc or ""))
        b_room_desc_tokens: dict[str, set[str]] = {}
        for room_name, desc in (
            db.query(LineItem.room, LineItem.description)
            .filter(LineItem.run_id == run_id, LineItem.doc == "B")
            .all()
        ):
            if room_name not in b_room_desc_tokens:
                b_room_desc_tokens[room_name] = set()
            b_room_desc_tokens[room_name].update(_tokens(desc or ""))
        room_group_b_by_room_a, room_group_component_count = _build_room_group_candidates(
            room_links_all,
            rooms_a,
            all_rooms_b,
            min_confidence=max(0.40, min_room_confidence - 0.18),
        )

        # clear previous matches
        db.execute(delete(Match).where(Match.run_id == run_id))
        db.commit()

        inserted = 0
        first_pass_total_evaluated = 0
        first_pass_uncertain_count = 0
        second_pass_reviewed_count = 0
        second_pass_rooms_invoked = 0
        nugget_count = 0
        status_counts = {"green": 0, "orange": 0, "blue": 0, "nugget": 0}
        matched_a_ids_seen: set[int] = set()
        critical_blue_items: list[dict[str, object]] = []
        globally_used_b_ids: set[int] = set()

        # ── Phase 1: pre-fetch all room data (sequential DB reads, no LLM) ──────────
        room_preps: list[dict] = []
        for room_a in rooms_a:
            room_links = links_by_room_a.get(room_a, [])
            a_items = (
                db.query(LineItem)
                .filter(LineItem.run_id == run_id, LineItem.doc == "A", LineItem.room == room_a)
                .order_by(LineItem.page, LineItem.id)
                .all()
            )
            if not a_items:
                continue

            # Keep room coverage even when map confidence is slightly below threshold.
            sorted_links = sorted(room_links, key=lambda lnk: float(lnk.confidence or 0.0), reverse=True)
            strong_links = [lnk for lnk in sorted_links if float(lnk.confidence or 0.0) >= min_room_confidence]
            room_map_mode = "strong"
            if strong_links:
                active_links = strong_links
            elif sorted_links and float(sorted_links[0].confidence or 0.0) >= max(0.45, min_room_confidence - 0.20):
                top_conf = float(sorted_links[0].confidence or 0.0)
                active_links = [
                    lnk
                    for lnk in sorted_links
                    if float(lnk.confidence or 0.0) >= max(0.45, top_conf - 0.12)
                ][:3]
                room_map_mode = "low-confidence-fallback"
            else:
                active_links = []
                room_map_mode = "no-strong-link-fallback"

            candidate_room_bs = list(dict.fromkeys(lnk.room_b for lnk in active_links))
            grouped_room_bs = room_group_b_by_room_a.get(room_a, [])
            if grouped_room_bs:
                for room_b_group in grouped_room_bs:
                    if room_b_group not in candidate_room_bs:
                        candidate_room_bs.append(room_b_group)
                if room_map_mode == "no-strong-link-fallback":
                    room_map_mode = "room-group-fallback"

            if not candidate_room_bs:
                soft_candidates = _soft_room_candidates(
                    room_a,
                    all_rooms_b,
                    a_room_desc_tokens,
                    b_room_desc_tokens,
                    limit=6,
                )
                for room_b_candidate in soft_candidates:
                    if room_b_candidate not in candidate_room_bs:
                        candidate_room_bs.append(room_b_candidate)
                if candidate_room_bs:
                    room_map_mode = "name-profile-fallback"

            if not candidate_room_bs and all_rooms_b:
                candidate_room_bs = list(all_rooms_b)
                room_map_mode = "all-rooms-fallback"

            expanded_room_bs = _expand_candidate_rooms(
                candidate_room_bs,
                all_rooms_b,
                b_room_desc_tokens,
            )
            if not expanded_room_bs:
                expanded_room_bs = list(candidate_room_bs)
            room_alias_expanded = len(expanded_room_bs) > len(candidate_room_bs)
            room_b = ", ".join(expanded_room_bs)

            b_items = (
                db.query(LineItem)
                .filter(
                    LineItem.run_id == run_id,
                    LineItem.doc == "B",
                    LineItem.room.in_(expanded_room_bs),
                )
                .order_by(LineItem.page, LineItem.id)
                .all()
            )

            if not b_items:
                soft_candidates = _soft_room_candidates(
                    room_a,
                    all_rooms_b,
                    a_room_desc_tokens,
                    b_room_desc_tokens,
                    limit=10,
                )
                if soft_candidates:
                    room_map_mode = "global-room-fallback"
                    room_b = ", ".join(soft_candidates)
                    b_items = (
                        db.query(LineItem)
                        .filter(
                            LineItem.run_id == run_id,
                            LineItem.doc == "B",
                            LineItem.room.in_(soft_candidates),
                        )
                        .order_by(LineItem.page, LineItem.id)
                        .all()
                    )

            if not b_items:
                room_preps.append({
                    "room_a": room_a, "room_b": room_b, "a_items": a_items,
                    "b_items": [], "a_dicts": [], "b_dicts": [], "b_by_id": {},
                    "room_map_mode": room_map_mode, "room_alias_expanded": False,
                    "llm_needed": False,
                })
                continue

            # Metadata-aware matching payloads (description + total + key metadata).
            a_dicts = [
                {
                    "id": x.id,
                    "room": x.room,
                    "description": x.description,
                    "amount": x.amount,
                    "quantity": x.quantity,
                    "unit": x.unit,
                    "unit_price": x.unit_price,
                }
                for x in a_items
            ]
            b_dicts = [
                {
                    "id": x.id,
                    "room": x.room,
                    "description": x.description,
                    "amount": x.amount,
                    "quantity": x.quantity,
                    "unit": x.unit,
                    "unit_price": x.unit_price,
                }
                for x in b_items
            ]
            b_by_id = {x.id: x for x in b_items}
            room_preps.append({
                "room_a": room_a, "room_b": room_b,
                "a_items": a_items, "b_items": b_items,
                "a_dicts": a_dicts, "b_dicts": b_dicts, "b_by_id": b_by_id,
                "room_map_mode": room_map_mode, "room_alias_expanded": room_alias_expanded,
                "llm_needed": True,
            })

        # ── Phase 2: LLM passes for all rooms in parallel ────────────────────────────
        llm_results: dict[int, dict] = {}
        llm_needed_indices = [(i, p) for i, p in enumerate(room_preps) if p["llm_needed"]]
        if llm_needed_indices:
            with ThreadPoolExecutor(max_workers=max(1, min(8, len(llm_needed_indices)))) as ex:
                futures = {
                    ex.submit(
                        _run_llm_for_room,
                        room_a=prep["room_a"],
                        room_b=prep["room_b"],
                        a_dicts=prep["a_dicts"],
                        b_dicts=prep["b_dicts"],
                        first_pass_unsure_conf=first_pass_unsure_conf,
                        first_pass_unsure_null_conf=first_pass_unsure_null_conf,
                        first_pass_scope_unsure_conf=first_pass_scope_unsure_conf,
                        force_review_on_null=force_review_on_null,
                    ): i
                    for i, prep in llm_needed_indices
                }
                for fut in as_completed(futures):
                    llm_results[futures[fut]] = fut.result()

        # ── Phase 3: assignment + DB writes (sequential, maintains globally_used_b_ids) ──
        for i, prep in enumerate(room_preps):
            room_a = prep["room_a"]
            room_b = prep["room_b"]
            a_items = prep["a_items"]
            b_items = prep["b_items"]
            b_by_id = prep["b_by_id"]
            room_map_mode = prep["room_map_mode"]
            room_alias_expanded = prep["room_alias_expanded"]

            if not prep["llm_needed"]:
                for a_item in a_items:
                    db.add(
                        Match(
                            run_id=run_id,
                            room_a=room_a,
                            room_b=room_b,
                            item_a_id=a_item.id,
                            item_b_id=None,
                            status="blue",
                            similarity=0.0,
                            rationale="mapped room has no Doc B items after fallback expansion",
                        )
                    )
                    inserted += 1
                    status_counts["blue"] = int(status_counts.get("blue", 0)) + 1
                    matched_a_ids_seen.add(a_item.id)
                continue

            llm_result = llm_results[i]
            first_best_for_a = llm_result["first_best_for_a"]
            first_critical_blue = llm_result["first_critical_blue"]
            second_best_for_a = llm_result["second_best_for_a"]
            second_critical_blue = llm_result["second_critical_blue"]
            first_pass_total_evaluated += llm_result["first_pass_total_evaluated"]
            first_pass_uncertain_count += llm_result["first_pass_uncertain_count"]
            second_pass_reviewed_count += llm_result["second_pass_reviewed_count"]
            second_pass_rooms_invoked += llm_result["second_pass_rooms_invoked"]

            # ── Simple greedy one-to-one assignment ──────────────────────────────────
            # 1. For each A item pick best proposal (second-pass > first-pass).
            # 2. Sort all proposals by confidence desc and assign greedily —
            #    first claim on a B item wins; conflicts resolved by confidence.
            # 3. Classify deterministically: scope_same + metadata ±2%.

            proposals: list[tuple[float, int, int | None, bool, str]] = []
            # (confidence, a_id, b_id_or_none, scope_same, rationale)
            for a_item in a_items:
                a_id = a_item.id
                second = second_best_for_a.get(a_id)
                first  = first_best_for_a.get(a_id)
                # Prefer second pass; if second pass returned null but first had a confident match, keep first.
                if second is not None:
                    b_id, scope_same, conf, rationale = second
                    if b_id is None and first is not None and first[0] is not None and float(first[2]) >= 0.55:
                        b_id, scope_same, conf, rationale = first
                        rationale = f"[kept pass1 over pass2-null] {rationale}"
                elif first is not None:
                    b_id, scope_same, conf, rationale = first
                else:
                    b_id, scope_same, conf, rationale = None, False, 0.0, "no proposal"
                proposals.append((float(conf), a_id, b_id, bool(scope_same), rationale or ""))

            # Sort highest confidence first so the best pairs claim B items first.
            proposals.sort(key=lambda x: x[0], reverse=True)

            assigned_for_a: dict[int, dict] = {}
            for conf, a_id, b_id, scope_same, rationale in proposals:
                if b_id is not None and b_id in b_by_id and b_id not in globally_used_b_ids:
                    globally_used_b_ids.add(b_id)
                    assigned_for_a[a_id] = {"item_b_id": b_id, "scope_same": scope_same,
                                            "confidence": conf, "rationale": rationale}
                else:
                    # B already taken or null — mark unmatched
                    if b_id is not None and b_id in globally_used_b_ids:
                        rationale = f"{rationale} | B={b_id} already claimed by higher-confidence pair"
                    assigned_for_a[a_id] = {"item_b_id": None, "scope_same": False,
                                            "confidence": conf, "rationale": rationale}

            # ── Persist: classify each assignment and write to DB ────────────────────
            for a_item in a_items:
                a_id = a_item.id
                result = assigned_for_a.get(a_id, {"item_b_id": None, "scope_same": False,
                                                    "confidence": 0.0, "rationale": "no proposal"})
                chosen_b = result["item_b_id"]
                scope_same = result["scope_same"]
                confidence = float(result["confidence"])
                rationale = result["rationale"]

                if chosen_b is None:
                    # No match from LLM → Blue
                    status = "blue"
                else:
                    b_item_matched = b_by_id[chosen_b]
                    # scope_same=True means LLM confirmed same underlying task → orange or green
                    # scope_same=False means LLM matched but was uncertain → orange
                    metadata_within_tol, _, mismatch_fields = _metadata_tolerance_result(
                        a_item, b_item_matched, tol=green_amount_tol,
                    )
                    if scope_same and metadata_within_tol:
                        status = "green"
                    else:
                        status = "orange"
                    if mismatch_fields:
                        rationale = f"{rationale} | metadata differs: {', '.join(mismatch_fields)}"

                if status == "blue":
                    llm_critical = second_critical_blue.get(a_id, first_critical_blue.get(a_id, False))
                    if llm_critical:
                        rationale = f"{rationale} | [CRITICAL_BLUE] high-priority JDR-only scope"
                        critical_blue_items.append({
                            "item_a_id": int(a_item.id),
                            "room_a": a_item.room,
                            "page": int(a_item.page),
                            "description": (a_item.description or "")[:180],
                        })

                db.add(
                    Match(
                        run_id=run_id,
                        room_a=room_a,
                        room_b=room_b,
                        item_a_id=a_id,
                        item_b_id=chosen_b,
                        status=status,
                        similarity=confidence,
                        rationale=rationale[:1500],
                    )
                )
                inserted += 1
                status_counts[status] = int(status_counts.get(status, 0)) + 1
                matched_a_ids_seen.add(a_id)

        # Add insurance-only nuggets (Doc B items not matched to any A item).
        all_b_items = (
            db.query(LineItem)
            .filter(LineItem.run_id == run_id, LineItem.doc == "B")
            .order_by(LineItem.page, LineItem.id)
            .all()
        )
        seen_nugget_sigs: set[tuple[str, str, float | None]] = set()
        for b_item in all_b_items:
            if b_item.id in globally_used_b_ids:
                continue
            sig = _nugget_signature(b_item.room, b_item.description or "", b_item.amount)
            if sig in seen_nugget_sigs:
                continue
            seen_nugget_sigs.add(sig)
            mapped_room_a = best_room_a_for_room_b.get(b_item.room, "(unmapped)")
            rationale = (
                f'Insurance-only nugget: "{(b_item.description or "").strip()}" '
                f'at {b_item.amount if b_item.amount is not None else "unknown amount"} '
                "appears in insurance but has no matched JDR line item."
            )
            db.add(
                Match(
                    run_id=run_id,
                    room_a=mapped_room_a,
                    room_b=b_item.room,
                    item_a_id=None,
                    item_b_id=b_item.id,
                    status="nugget",
                    similarity=0.0,
                    rationale=rationale[:1500],
                )
            )
            inserted += 1
            nugget_count += 1
            status_counts["nugget"] = int(status_counts.get("nugget", 0)) + 1

        db.commit()
        total_a_items = (
            db.query(func.count(LineItem.id))
            .filter(LineItem.run_id == run_id, LineItem.doc == "A")
            .scalar()
            or 0
        )
        coverage_audit = {
            "total_a_items": int(total_a_items),
            "matched_a_rows": int(len(matched_a_ids_seen)),
            "missing_a_rows": int(max(0, int(total_a_items) - len(matched_a_ids_seen))),
            "critical_blue_count": int(len(critical_blue_items)),
            "critical_blue_examples": critical_blue_items[:8],
        }
        elapsed_ms = int((perf_counter() - t0) * 1000)
        return {
            "run_id": run_id,
            "matches_inserted": inserted,
            "first_pass_model": settings.matching_first_pass_model,
            "second_pass_models": settings.matching_second_pass_model_list,
            "first_pass_total_evaluated": first_pass_total_evaluated,
            "first_pass_uncertain_count": first_pass_uncertain_count,
            "second_pass_reviewed_count": second_pass_reviewed_count,
            "second_pass_rooms_invoked": second_pass_rooms_invoked,
            "nugget_count": nugget_count,
            "blue_guardrail": "null/low-conf forced to pass2; rescue+reconcile before blue",
            "status_counts": status_counts,
            "coverage_audit": coverage_audit,
            "llm_telemetry": get_match_telemetry(),
            "elapsed_ms": elapsed_ms,
            "matching_mode": "semantic_plus_metadata_2pct",
            "room_group_component_count": int(room_group_component_count),
            "room_group_room_a_coverage": int(len(room_group_b_by_room_a)),
        }

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
