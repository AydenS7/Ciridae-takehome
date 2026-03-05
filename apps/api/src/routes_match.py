from __future__ import annotations

import heapq
import re
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
    reset_match_telemetry,
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


def _metadata_tolerance_result(
    a_item: LineItem,
    b_item: LineItem,
    *,
    tol: float,
) -> tuple[bool, bool, list[str]]:
    """
    Returns:
    - metadata_within_tol: all key fields are aligned within tolerance
    - amount_within_tol: total/amount exists on both and is within tolerance
    - mismatch_fields: key fields outside tolerance or missing on one side
    """
    mismatch_fields: list[str] = []

    def _check_numeric(name: str, a_val: float | None, b_val: float | None) -> None:
        if a_val is None and b_val is None:
            return
        if (a_val is None) != (b_val is None):
            mismatch_fields.append(name)
            return
        if _pct_diff(a_val, b_val) > tol:
            mismatch_fields.append(name)

    _check_numeric("amount", a_item.amount, b_item.amount)
    _check_numeric("quantity", a_item.quantity, b_item.quantity)
    _check_numeric("unit_price", a_item.unit_price, b_item.unit_price)

    a_unit = _canonical_unit(a_item.unit)
    b_unit = _canonical_unit(b_item.unit)
    if a_unit or b_unit:
        if not a_unit or not b_unit or a_unit != b_unit:
            mismatch_fields.append("unit")

    metadata_within_tol = len(mismatch_fields) == 0
    amount_within_tol = (
        a_item.amount is not None
        and b_item.amount is not None
        and _pct_diff(a_item.amount, b_item.amount) <= tol
    )
    return metadata_within_tol, amount_within_tol, mismatch_fields


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

@router.post("/{run_id}/match")
def match_run(run_id: str, min_room_confidence: float = 0.6, min_pair_confidence: float | None = None):
    """
    LLM-only pairing per mapped room, deterministic classification:
    - comparisons use semantic name/description + key metadata (amount/quantity/unit/unit_price)
    - blue: A-only (no B match)
    - green: semantic scope match AND key metadata within ±2%
    - orange: semantic scope match but one or more key metadata fields differ beyond ±2%
    - nugget: insurance-only (B present with no matched A)
    """
    t0 = perf_counter()
    reset_match_telemetry()
    with SessionLocal() as db:
        effective_min_pair_conf = (
            settings.matching_accept_confidence if min_pair_confidence is None else float(min_pair_confidence)
        )
        first_pass_unsure_conf = float(settings.matching_first_pass_unsure_confidence)
        first_pass_unsure_null_conf = float(settings.matching_first_pass_unsure_null_confidence)
        first_pass_scope_unsure_conf = float(settings.matching_first_pass_scope_unsure_confidence)
        force_review_on_null = bool(settings.matching_force_review_on_null)
        green_amount_tol = float(settings.matching_green_amount_tolerance_pct)
        fallback_min_score = float(settings.matching_fallback_similarity_threshold)
        reconcile_min_score = float(settings.matching_reconcile_similarity_threshold)
        reconcile_scope_desc_sim = float(settings.matching_reconcile_scope_similarity_threshold)
        scope_sim_threshold = float(settings.matching_scope_similarity_threshold)
        scope_overlap_threshold = float(settings.matching_scope_overlap_threshold)
        assignment_unmatched_threshold = float(settings.matching_assignment_unmatched_threshold)
        blue_guard_unused_threshold = float(settings.matching_blue_guard_unused_threshold)
        blue_guard_consumed_threshold = float(settings.matching_blue_guard_consumed_threshold)
        rescue_green_price_tol = float(settings.rescue_green_price_tol)
        rescue_orange_price_tol = float(settings.rescue_orange_price_tol)
        rescue_min_desc_sim_green = float(settings.rescue_min_desc_sim_green)
        rescue_min_desc_sim_orange = float(settings.rescue_min_desc_sim_orange)

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

            first_pass = propose_matches_for_room(room_a, room_b, a_dicts, b_dicts)
            first_best_for_a = _best_pairs_by_a(first_pass)
            first_critical_blue = _critical_blue_by_a(first_pass)
            first_pass_total_evaluated += len(a_items)

            first_pass_unsure_ids: set[int] = set()
            for a_item in a_items:
                choice = first_best_for_a.get(a_item.id)
                if choice is None:
                    first_pass_unsure_ids.add(a_item.id)
                    continue
                item_b_id, scope_same, conf, _ = choice
                if item_b_id is None and force_review_on_null:
                    first_pass_unsure_ids.add(a_item.id)
                    continue
                if conf < first_pass_unsure_conf:
                    first_pass_unsure_ids.add(a_item.id)
                    continue
                if item_b_id is None and conf < first_pass_unsure_null_conf:
                    first_pass_unsure_ids.add(a_item.id)
                    continue
                if (not bool(scope_same)) and conf < first_pass_scope_unsure_conf:
                    first_pass_unsure_ids.add(a_item.id)
            first_pass_uncertain_count += len(first_pass_unsure_ids)

            uncertain_a_ids = set(first_pass_unsure_ids)

            second_best_for_a: dict[int, tuple[int | None, bool, float, str]] = {}
            second_critical_blue: dict[int, bool] = {}
            if uncertain_a_ids:
                uncertain_a_dicts = [x for x in a_dicts if int(x["id"]) in uncertain_a_ids]
                first_pass_uncertain_by_a = {
                    a_id: first_best_for_a[a_id]
                    for a_id in uncertain_a_ids
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
                second_pass_reviewed_count += len(uncertain_a_ids)
                second_pass_rooms_invoked += 1

            # Build one candidate per A item (prefer second-pass decision when available).
            candidate_for_a: dict[int, dict] = {}
            for a_item in a_items:
                a_id = a_item.id
                first_choice = first_best_for_a.get(a_id)
                second_choice = second_best_for_a.get(a_id)
                chosen = second_best_for_a.get(a_id) or first_best_for_a.get(a_id)
                second_overrode_with_null = False
                chosen_via_quality = False
                if second_choice is not None and second_choice[0] is None and first_choice is not None and first_choice[0] is not None:
                    keep_first_candidate = float(first_choice[2]) >= max(0.32, effective_min_pair_conf - 0.05)
                    if keep_first_candidate:
                        chosen = first_choice
                        second_overrode_with_null = True
                if (
                    first_choice is not None
                    and second_choice is not None
                    and first_choice[0] is not None
                    and second_choice[0] is not None
                    and first_choice[0] != second_choice[0]
                ):
                    q_first = _choice_quality(
                        a_item=a_item,
                        choice=first_choice,
                        b_by_id=b_by_id,
                        green_amount_tol=green_amount_tol,
                        scope_sim_threshold=scope_sim_threshold,
                        scope_overlap_threshold=scope_overlap_threshold,
                    )
                    q_second = _choice_quality(
                        a_item=a_item,
                        choice=second_choice,
                        b_by_id=b_by_id,
                        green_amount_tol=green_amount_tol,
                        scope_sim_threshold=scope_sim_threshold,
                        scope_overlap_threshold=scope_overlap_threshold,
                    )
                    if q_first > q_second + 0.05:
                        chosen = first_choice
                        chosen_via_quality = True
                    elif q_second > q_first + 0.05:
                        chosen = second_choice
                        chosen_via_quality = True
                rationales: list[str] = []
                if room_map_mode != "strong":
                    mode_reason = {
                        "low-confidence-fallback": (
                            "room-map fallback: included lower-confidence mapped room candidates to reduce false blue"
                        ),
                        "room-group-fallback": (
                            "room-group fallback: expanded to B-rooms connected through many-to-many room groups"
                        ),
                        "name-profile-fallback": (
                            "room similarity fallback: selected candidate B-rooms by room-name + room-profile similarity"
                        ),
                        "global-room-fallback": (
                            "global room fallback: mapped room had no B-items; broadened to top similar B-rooms"
                        ),
                        "all-rooms-fallback": (
                            "all-rooms fallback: no reliable mapping candidates; searched across all insurance rooms"
                        ),
                        "no-strong-link-fallback": (
                            "room-map fallback: no strong room links; kept best available candidates"
                        ),
                    }.get(
                        room_map_mode,
                        "room-map fallback: used alternate room candidates to avoid false blue",
                    )
                    rationales.append(mode_reason)
                if room_alias_expanded:
                    rationales.append(
                        "room-alias expansion: included closely related room names from mapped insurance rooms"
                    )
                if first_choice is not None:
                    fp_b, fp_scope, fp_conf, fp_reason = first_choice
                    fp_core = f"pass1 target={fp_b} conf={fp_conf:.2f} scope_same={bool(fp_scope)}"
                    if fp_reason:
                        fp_core = f"{fp_core}: {fp_reason}"
                    rationales.append(fp_core)
                if second_choice is not None:
                    sp_b, sp_scope, sp_conf, sp_reason = second_choice
                    sp_core = f"pass2-review target={sp_b} conf={sp_conf:.2f} scope_same={bool(sp_scope)}"
                    if sp_reason:
                        sp_core = f"{sp_core}: {sp_reason}"
                    rationales.append(sp_core)
                if second_overrode_with_null:
                    rationales.append("pass2 returned null; retained plausible pass1 candidate to avoid false blue")
                if chosen_via_quality and first_choice is not None and second_choice is not None:
                    rationales.append(
                        f"quality arbitration selected target={chosen[0]} over pass1={first_choice[0]} pass2={second_choice[0]}"
                    )

                if chosen is None:
                    candidate_for_a[a_id] = {
                        "item_b_id": None,
                        "scope_same": False,
                        "confidence": 0.0,
                        "source": "none",
                        "rationales": rationales + ["No model proposal returned for this item."],
                    }
                    continue
                c_b, c_scope, c_conf, c_reason = chosen
                source = "pass2-ensemble" if a_id in second_best_for_a else "pass1"
                if c_reason:
                    rationales.append(f"{source}: {c_reason}".strip(": "))
                else:
                    rationales.append(source)
                candidate_for_a[a_id] = {
                    "item_b_id": c_b,
                    "scope_same": bool(c_scope),
                    "confidence": float(c_conf),
                    "source": source,
                    "rationales": rationales,
                }

            # Lock very-high-confidence near-exact pairs before global optimization.
            locked_by_b: dict[int, tuple[float, int]] = {}
            for a_item in a_items:
                a_id = a_item.id
                cand = candidate_for_a.get(a_id)
                if cand is None:
                    continue
                b_id = cand.get("item_b_id")
                if not isinstance(b_id, int) or b_id not in b_by_id or b_id in globally_used_b_ids:
                    continue
                semantic_locked, desc_sim_locked, desc_overlap_locked = _semantic_same(
                    a_item.description or "",
                    b_by_id[b_id].description or "",
                    scope_same_hint=bool(cand.get("scope_same")),
                    scope_sim_threshold=scope_sim_threshold,
                    scope_overlap_threshold=scope_overlap_threshold,
                )
                if not semantic_locked:
                    continue
                conf_locked = float(cand.get("confidence") or 0.0)
                within_tol_locked = (
                    _pct_diff(a_item.amount, b_by_id[b_id].amount) <= green_amount_tol
                    if (a_item.amount is not None and b_by_id[b_id].amount is not None)
                    else False
                )
                if conf_locked < 0.86:
                    continue
                if not (within_tol_locked or (desc_sim_locked >= 0.74 and desc_overlap_locked >= 0.78)):
                    continue
                lock_score = (
                    conf_locked
                    + (0.25 if within_tol_locked else 0.0)
                    + (0.20 * desc_sim_locked)
                    + (0.20 * desc_overlap_locked)
                )
                prev = locked_by_b.get(b_id)
                if prev is None or lock_score > prev[0]:
                    locked_by_b[b_id] = (lock_score, a_id)

            preassigned_for_a: dict[int, dict] = {}
            prelocked_b_ids: set[int] = set()
            for b_id, (_, a_id) in locked_by_b.items():
                cand = candidate_for_a[a_id]
                preassigned_for_a[a_id] = {
                    "item_b_id": b_id,
                    "scope_same": bool(cand.get("scope_same", True)),
                    "confidence": float(cand.get("confidence") or 0.0),
                    "rationales": list(cand.get("rationales") or [])
                    + [f"prelocked high-confidence exact pair B={b_id}"],
                    "force_status": None,
                }
                prelocked_b_ids.add(b_id)

            globally_used_b_ids |= prelocked_b_ids

            # Build multi-candidate score graph per A item, then optimize one-to-one assignment globally.
            available_b_ids = [b.id for b in b_items if b.id not in globally_used_b_ids]
            candidate_scores_by_a: dict[int, dict[int, float]] = {}
            candidate_meta_by_a: dict[int, dict[int, dict[str, float | bool | str]]] = {}

            for a_item in a_items:
                a_id = a_item.id
                if a_id in preassigned_for_a:
                    continue
                scores_for_a: dict[int, float] = {}
                meta_for_a: dict[int, dict[str, float | bool | str]] = {}
                first_choice = first_best_for_a.get(a_id)
                second_choice = second_best_for_a.get(a_id)

                def add_choice_edge(choice: tuple[int | None, bool, float, str] | None, source_label: str) -> None:
                    if choice is None:
                        return
                    b_id, scope_hint, conf, _ = choice
                    if b_id is None or b_id not in b_by_id or b_id not in available_b_ids:
                        return
                    b_item = b_by_id[b_id]
                    pair_score, _desc_sim_raw, amt_sim = _pair_similarity(a_item, b_item)
                    semantic, desc_sim, desc_overlap = _semantic_same(
                        a_item.description or "",
                        b_item.description or "",
                        scope_same_hint=bool(scope_hint),
                        scope_sim_threshold=scope_sim_threshold,
                        scope_overlap_threshold=scope_overlap_threshold,
                    )
                    within_tol = (
                        _pct_diff(a_item.amount, b_item.amount) <= green_amount_tol
                        if (a_item.amount is not None and b_item.amount is not None)
                        else False
                    )
                    edge_score = pair_score + (0.18 if semantic else 0.0) + (0.20 if within_tol else 0.0)
                    edge_score += min(0.18, 0.10 * float(conf))
                    edge_score += (0.05 if source_label == "pass2" else 0.03)
                    edge_score = max(edge_score, 0.0)
                    prev = scores_for_a.get(b_id)
                    if prev is None or edge_score > prev:
                        scores_for_a[b_id] = edge_score
                        meta_for_a[b_id] = {
                            "semantic": semantic,
                            "confidence": max(float(conf), pair_score),
                            "desc_sim": desc_sim,
                            "desc_overlap": desc_overlap,
                            "amount_sim": amt_sim,
                            "source": source_label,
                        }

                add_choice_edge(first_choice, "pass1")
                add_choice_edge(second_choice, "pass2")

                ranked_edges: list[tuple[float, int, bool, float, float, float]] = []
                for b_id in available_b_ids:
                    b_item = b_by_id[b_id]
                    pair_score, desc_sim, amt_sim = _pair_similarity(a_item, b_item)
                    semantic, _, desc_overlap = _semantic_same(
                        a_item.description or "",
                        b_item.description or "",
                        scope_same_hint=False,
                        scope_sim_threshold=scope_sim_threshold,
                        scope_overlap_threshold=scope_overlap_threshold,
                    )
                    if pair_score < max(0.26, reconcile_min_score - 0.18) and not semantic:
                        continue
                    within_tol = (
                        _pct_diff(a_item.amount, b_item.amount) <= green_amount_tol
                        if (a_item.amount is not None and b_item.amount is not None)
                        else False
                    )
                    score = pair_score + (0.15 if semantic else 0.0) + (0.18 if within_tol else 0.0)
                    ranked_edges.append((score, b_id, semantic, desc_sim, desc_overlap, amt_sim))

                ranked_edges.sort(key=lambda x: x[0], reverse=True)
                for score, b_id, semantic, desc_sim, desc_overlap, amt_sim in ranked_edges[:8]:
                    prev = scores_for_a.get(b_id)
                    if prev is None or score > prev:
                        scores_for_a[b_id] = score
                        meta_for_a[b_id] = {
                            "semantic": semantic,
                            "confidence": score,
                            "desc_sim": desc_sim,
                            "desc_overlap": desc_overlap,
                            "amount_sim": amt_sim,
                            "source": "deterministic",
                        }

                candidate_scores_by_a[a_id] = scores_for_a
                candidate_meta_by_a[a_id] = meta_for_a

            a_ids_to_optimize = [a.id for a in a_items if a.id not in preassigned_for_a]
            optimized_assignment = _optimize_room_assignment(
                a_ids=a_ids_to_optimize,
                b_ids=available_b_ids,
                score_by_a=candidate_scores_by_a,
                unmatched_threshold=max(0.25, min(0.95, assignment_unmatched_threshold)),
            )

            assigned_for_a: dict[int, dict] = {}
            for a_item in a_items:
                a_id = a_item.id
                if a_id in preassigned_for_a:
                    assigned_for_a[a_id] = preassigned_for_a[a_id]
                    continue
                cand = candidate_for_a[a_id]
                item_b_id = cand["item_b_id"]
                confidence = float(cand["confidence"])
                scope_same = bool(cand["scope_same"])
                source = str(cand.get("source") or "none")
                rationales = list(cand["rationales"])
                force_status: str | None = None

                chosen_b = optimized_assignment.get(a_id)
                if chosen_b is not None and chosen_b in b_by_id and chosen_b not in globally_used_b_ids:
                    meta = candidate_meta_by_a.get(a_id, {}).get(chosen_b, {})
                    if chosen_b == item_b_id:
                        # Optimizer confirmed LLM's chosen B — trust LLM's scope_same
                        scope_same = bool(meta.get("semantic", False)) or bool(cand.get("scope_same", False))
                    else:
                        # Optimizer picked a different B than LLM suggested — use text-based semantic
                        scope_same = bool(meta.get("semantic", scope_same))
                    confidence = max(confidence, float(meta.get("confidence", 0.0)))
                    rationales.append(
                        f'global assignment selected B={chosen_b} source={meta.get("source", "unknown")}'
                    )
                    globally_used_b_ids.add(chosen_b)
                else:
                    chosen_b = None
                    if (
                        item_b_id is not None
                        and item_b_id in b_by_id
                        and item_b_id not in globally_used_b_ids
                        and (
                            confidence >= effective_min_pair_conf
                            or source == "pass2-ensemble"
                        )
                    ):
                        chosen_b = item_b_id
                        globally_used_b_ids.add(chosen_b)
                    else:
                        if item_b_id is not None and item_b_id in globally_used_b_ids:
                            rationales.append("candidate B item already used by a higher-confidence pair")
                        if (
                            item_b_id is not None
                            and confidence < effective_min_pair_conf
                            and source != "pass2-ensemble"
                        ):
                            rationales.append("candidate pair confidence below threshold")

                    if chosen_b is None:
                        # Deterministic fallback prevents weak blue misclassifications.
                        fb_b, fb_score, fb_reason = _fallback_match(
                            a_item,
                            b_items,
                            globally_used_b_ids,
                            min_score=fallback_min_score,
                        )
                        if fb_b is not None:
                            chosen_b = fb_b
                            globally_used_b_ids.add(chosen_b)
                            scope_same = fb_score >= 0.62
                            confidence = max(confidence, fb_score)
                            if fb_reason:
                                rationales.append(fb_reason)
                        else:
                            # Last chance lexical rescue for cases where model reasoning is too strict.
                            best_b: LineItem | None = None
                            best_desc_overlap = 0.0
                            for b_item in b_items:
                                if b_item.id in globally_used_b_ids:
                                    continue
                                overlap = _token_overlap_on_shorter(a_item.description or "", b_item.description or "")
                                if overlap > best_desc_overlap:
                                    best_desc_overlap = overlap
                                    best_b = b_item
                            if best_b is not None and best_desc_overlap >= 0.82:
                                chosen_b = best_b.id
                                globally_used_b_ids.add(chosen_b)
                                scope_same = True
                                confidence = max(confidence, min(0.70, best_desc_overlap))
                                rationales.append(
                                    f"lexical rescue overlap={best_desc_overlap:.2f} to prevent false blue"
                                )

                if chosen_b is not None and chosen_b in b_by_id:
                    semantic_chosen, desc_sim_chosen, desc_overlap_chosen = _semantic_same(
                        a_item.description or "",
                        b_by_id[chosen_b].description or "",
                        scope_same_hint=scope_same,
                        scope_sim_threshold=scope_sim_threshold,
                        scope_overlap_threshold=scope_overlap_threshold,
                    )
                    if not semantic_chosen and not scope_same:
                        # Only release if LLM also said NOT same scope AND text sim is low.
                        # If scope_same=True, keep the pair — persist loop will classify as orange.
                        if chosen_b in globally_used_b_ids:
                            globally_used_b_ids.remove(chosen_b)
                        rationales.append(
                            "released non-semantic candidate "
                            f"B={chosen_b} (desc_sim={desc_sim_chosen:.2f}, overlap={desc_overlap_chosen:.2f})"
                        )
                        chosen_b = None
                        scope_same = False

                if chosen_b is None:
                    # Blue guardrail: only keep blue when no strong semantic candidate exists.
                    rescue_sim_threshold = max(0.34, scope_sim_threshold - 0.08)
                    rescue_overlap_threshold = max(0.48, scope_overlap_threshold - 0.08)
                    best_unused: tuple[float, int, float, float] | None = None
                    best_any: tuple[float, int, bool, bool, float, float] | None = None
                    for b_item in b_items:
                        desc_sim = _token_jaccard(a_item.description or "", b_item.description or "")
                        desc_overlap = _token_overlap_on_shorter(a_item.description or "", b_item.description or "")
                        semantic = bool(
                            desc_sim >= rescue_sim_threshold
                            or desc_overlap >= rescue_overlap_threshold
                        )
                        semantic_strength = max(desc_sim, desc_overlap * 0.98)
                        if best_any is None or semantic_strength > best_any[0]:
                            best_any = (
                                semantic_strength,
                                b_item.id,
                                semantic,
                                b_item.id in globally_used_b_ids,
                                desc_sim,
                                desc_overlap,
                            )
                        if b_item.id in globally_used_b_ids:
                            continue
                        if not semantic:
                            continue
                        if best_unused is None or semantic_strength > best_unused[0]:
                            best_unused = (semantic_strength, b_item.id, desc_sim, desc_overlap)

                    if best_unused is not None:
                        score_u, b_id_u, desc_sim_u, desc_overlap_u = best_unused
                        if score_u >= max(0.32, blue_guard_unused_threshold):
                            chosen_b = b_id_u
                            scope_same = True
                            confidence = max(confidence, score_u)
                            globally_used_b_ids.add(chosen_b)
                            rationales.append(
                                "blue-guard reassigned to unused semantic candidate "
                                f"B={b_id_u} score={score_u:.2f} "
                                f"(desc_sim={desc_sim_u:.2f}, overlap={desc_overlap_u:.2f})"
                            )

                    if chosen_b is None and best_any is not None:
                        score_a, b_id_a, semantic_a, used_flag, desc_sim_a, desc_overlap_a = best_any
                        if (
                            semantic_a
                            and score_a >= max(0.44, blue_guard_consumed_threshold + 0.08)
                            and (desc_sim_a >= 0.40 or desc_overlap_a >= 0.50)
                        ):
                            force_status = "orange"
                            if used_flag:
                                rationales.append(
                                    "blue-guard: close semantic candidate "
                                    f"B={b_id_a} score={score_a:.2f} "
                                    f"(desc_sim={desc_sim_a:.2f}, overlap={desc_overlap_a:.2f}) "
                                    "already consumed by a higher-priority pair"
                                )

                assigned_for_a[a_id] = {
                    "item_b_id": chosen_b,
                    "scope_same": scope_same,
                    "confidence": confidence,
                    "rationales": rationales,
                    "force_status": force_status,
                }

            # Reconciliation pass: pair remaining A-only and B-only rows using deterministic similarity.
            unmatched_a_ids = [a.id for a in a_items if assigned_for_a[a.id]["item_b_id"] is None]
            remaining_b_ids = [b.id for b in b_items if b.id not in globally_used_b_ids]
            if unmatched_a_ids and remaining_b_ids:
                pair_candidates: list[tuple[float, float, float, int, int]] = []
                for a_item in a_items:
                    if a_item.id not in unmatched_a_ids:
                        continue
                    for b_id in remaining_b_ids:
                        b_item = b_by_id[b_id]
                        score, desc_sim, amt_sim = _pair_similarity(a_item, b_item)
                        if score < reconcile_min_score:
                            continue
                        pair_candidates.append((score, desc_sim, amt_sim, a_item.id, b_id))

                pair_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                used_a_ids: set[int] = set()
                used_b_ids_local: set[int] = set()
                for score, desc_sim, amt_sim, a_id, b_id in pair_candidates:
                    if a_id in used_a_ids or b_id in used_b_ids_local or b_id in globally_used_b_ids:
                        continue
                    used_a_ids.add(a_id)
                    used_b_ids_local.add(b_id)
                    globally_used_b_ids.add(b_id)

                    current = assigned_for_a[a_id]
                    revised_scope_same = bool(
                        desc_sim >= reconcile_scope_desc_sim
                        or (desc_sim >= 0.50 and amt_sim >= 0.75)
                    )
                    revised_conf = max(float(current["confidence"]), float(score))
                    revised_rationales = list(current["rationales"])
                    revised_rationales.append(
                        f"reconciled similarity={score:.2f} (desc={desc_sim:.2f}, amount={amt_sim:.2f})"
                    )
                    assigned_for_a[a_id] = {
                        "item_b_id": b_id,
                        "scope_same": revised_scope_same,
                        "confidence": revised_conf,
                        "rationales": revised_rationales,
                    }

            # Price-proximity rescue: catch any still-unmatched A items via price + desc signal.
            _price_proximity_rescue(
                a_items,
                b_by_id,
                assigned_for_a,
                globally_used_b_ids,
                green_price_tol=min(green_amount_tol, rescue_green_price_tol),
                orange_price_tol=rescue_orange_price_tol,
                min_desc_sim_green=rescue_min_desc_sim_green,
                min_desc_sim_orange=rescue_min_desc_sim_orange,
            )

            # Persist in source order for readability/debugging.
            for a_item in a_items:
                a_id = a_item.id
                result = assigned_for_a[a_id]
                chosen_b = result["item_b_id"]
                scope_same = result["scope_same"]
                confidence = float(result["confidence"])
                rationales = result["rationales"]
                force_status = result.get("force_status")

                a_amt = a_item.amount
                b_item_matched = b_by_id.get(chosen_b) if chosen_b is not None else None
                b_amt = b_item_matched.amount if b_item_matched is not None else None
                if force_status in {"orange", "green"}:
                    status = str(force_status)
                elif chosen_b is None:
                    status = "blue"
                else:
                    semantic_same, desc_sim, desc_overlap = _semantic_same(
                        a_item.description or "",
                        b_by_id[chosen_b].description or "",
                        scope_same_hint=scope_same,
                        scope_sim_threshold=scope_sim_threshold,
                        scope_overlap_threshold=scope_overlap_threshold,
                    )
                    metadata_within_tol = False
                    amount_within_tol = False
                    mismatch_fields: list[str] = []
                    if b_item_matched is not None:
                        metadata_within_tol, amount_within_tol, mismatch_fields = _metadata_tolerance_result(
                            a_item,
                            b_item_matched,
                            tol=green_amount_tol,
                        )
                    if mismatch_fields:
                        rationales.append(
                            f"metadata differs beyond ±{green_amount_tol*100:.0f}%: {', '.join(mismatch_fields)}"
                        )

                    if semantic_same or scope_same:
                        # semantic_same: text similarity confirms scope match
                        # scope_same: LLM explicitly confirmed same scope (trust even with low text overlap)
                        # Green requires BOTH scope confirmed AND all metadata within ±2% — 2% rule preserved.
                        status = "green" if (amount_within_tol and metadata_within_tol) else "orange"
                    else:
                        status = "blue"
                        rationales.append("non-semantic candidate downgraded to blue in final classification")
                        if chosen_b in globally_used_b_ids:
                            globally_used_b_ids.remove(chosen_b)
                        chosen_b = None

                if status == "blue":
                    # Use LLM's assessment (second-pass takes priority; fall back to first-pass)
                    llm_critical = second_critical_blue.get(a_id, first_critical_blue.get(a_id, False))
                    if llm_critical:
                        rationales.append(
                            "[CRITICAL_BLUE] LLM-flagged: high-priority JDR-only scope; explicit reviewer attention required"
                        )
                        critical_blue_items.append(
                            {
                                "item_a_id": int(a_item.id),
                                "room_a": a_item.room,
                                "page": int(a_item.page),
                                "description": (a_item.description or "")[:180],
                            }
                        )

                db.add(
                    Match(
                        run_id=run_id,
                        room_a=room_a,
                        room_b=room_b,
                        item_a_id=a_id,
                        item_b_id=chosen_b,
                        status=status,
                        similarity=confidence,
                        rationale=" | ".join(x for x in rationales if x)[:1500],
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
