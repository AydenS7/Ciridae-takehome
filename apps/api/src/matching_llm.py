"""LLM orchestration for room-scoped line-item matching and reviewer fallback."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Iterable

from .llm_client import get_client
from .llm_match_schemas import MatchPlan, ProposedPair
from .llm_utils import expand_model_variants, normalize_model_list
from .settings import settings

SYSTEM_PROMPT = """You match JDR (contractor, Doc A) line items to Insurance (Doc B) line items within the SAME room.

ACCURACY IS CRITICAL. Wrong matches (misidentifying which B item goes with an A item) cause incorrect green/orange/blue labels in the final report. Take your time. Be precise.

Item format: Doc A items typically start with "N." e.g. "1. Remove and replace drywall". Use this number to identify each item.

For EVERY A item, output exactly one ProposedPair — no omissions. Count your output before finishing.

Matching rules — follow in order:
1. Check exact_name_candidates_by_a[item_a_id] first — if present, that IS the match. Do not second-guess it.
2. Check near_name_candidates_by_a[item_a_id] next — close name = confirmed match.
3. If no candidates, find the closest-named B item in the room for the same underlying task.
4. Return null ONLY when the task genuinely does not appear in B for this room. Do NOT return null just because wording differs slightly — use semantic judgment.

scope_same:
- true  → same underlying repair task (use whenever names are close or candidates exist)
- false → matched but genuinely uncertain — use sparingly, only for truly ambiguous wording

Labor-only items: JDR estimates sometimes separate labor from material as distinct line items
(e.g. "Door prep - Labor only" alongside "Door hardware - material"). Insurance estimates
typically bundle labor into the combined work item without a separate "Labor only" line.
If an A item ends in "- Labor only" or "- labor" and the B side has no matching "Labor only"
item, match it to the B item covering the same underlying task (the one that would include
the labor). Use scope_same=true — the work IS covered, just not split out.

Specification awareness — scope_same = false when descriptions differ on:
- Material specs: "1/2 inch plywood" ≠ "5/8 inch plywood"; "1-coat" ≠ "2-coat"
- Quantities in description text: "2 windows" vs "1 window" is a scope difference
- Grade/type markers: e.g. "Type X drywall" ≠ "standard drywall"
Ignore minor wording variations (e.g. "1/2\"" vs "1/2 inch" are the same).

One-to-one: each B item may be used at most once. If two A items want the same B item, assign it to the closer name match and find the next best for the other.

For null matches: set critical_blue=true if the item is high-priority scope
(permits, electrical, safety testing, code compliance, hazards, engineering, liability-critical work).
"""

REVIEWER_SYSTEM_PROMPT = """You review uncertain first-pass matches between JDR (Doc A) and Insurance (Doc B) line items.

ACCURACY IS CRITICAL. A wrong match is worse than a missed match. Review carefully.

For every uncertain A item, output exactly one ProposedPair. Count your output before finishing.

Rules:
- If exact or near-name candidate exists, match it — the name IS the ground truth.
- Return null only when no B item in the room plausibly covers the same task — not just because wording differs.
- scope_same=true when same task; scope_same=false when specs differ (thickness, coat count, quantities in description, material grade) or description is genuinely ambiguous.
- Override first-pass null if a name candidate exists. Override first-pass match only if it's clearly wrong.
- One sentence rationale: why you confirmed or overrode the first pass.
- Labor-only items: if an A item says "- Labor only" and the first pass returned null, check
  whether a B item covers the same underlying task (insurance bundles labor in; no separate
  "Labor only" line is expected). If so, override to that B item with scope_same=true.

For null matches: set critical_blue=true for high-priority omitted scope
(permits, electrical, safety testing, code compliance, hazards, engineering, liability-critical work).
"""

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the",
    "and",
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

_MATCH_ALIAS_MAP: dict[str, str] = {
    "openai": "openai/gpt-4.1-mini",
    "chatgpt": "openai/gpt-4.1-mini",
    "gpt5": "openai/gpt-5",
    "gpt-5": "openai/gpt-5",
    "gpt5.3": "openai/gpt-5.3",
    "gpt-5.3": "openai/gpt-5.3",
    "anthropic": "anthropic/claude-3-5-sonnet-latest",
    "claude": "anthropic/claude-3-5-sonnet-latest",
    "gemini": "gemini/gemini-2.5-pro",
    "google": "gemini/gemini-2.5-pro",
}

FirstPassDecision = tuple[int | None, bool, float, str]

_MATCH_CALLS = 0
_MATCH_ATTEMPTS = 0
_MATCH_FALLBACK_SUCCESSES = 0
_MATCH_REVIEWER_FALLBACKS = 0
_MATCH_MODEL_USAGE: Counter[str] = Counter()


def reset_match_telemetry() -> None:
    global _MATCH_CALLS, _MATCH_ATTEMPTS, _MATCH_FALLBACK_SUCCESSES, _MATCH_REVIEWER_FALLBACKS, _MATCH_MODEL_USAGE
    _MATCH_CALLS = 0
    _MATCH_ATTEMPTS = 0
    _MATCH_FALLBACK_SUCCESSES = 0
    _MATCH_REVIEWER_FALLBACKS = 0
    _MATCH_MODEL_USAGE = Counter()


def get_match_telemetry() -> dict[str, Any]:
    return {
        "calls": int(_MATCH_CALLS),
        "attempts": int(_MATCH_ATTEMPTS),
        "fallback_successes": int(_MATCH_FALLBACK_SUCCESSES),
        "reviewer_fallbacks": int(_MATCH_REVIEWER_FALLBACKS),
        "models_used": dict(_MATCH_MODEL_USAGE),
    }


def _normalize_desc(text: str) -> str:
    s = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", (text or "").strip().lower())
    toks = [tok for tok in _TOKEN_RE.findall(s) if tok and tok not in _STOPWORDS]
    return " ".join(toks)


def _desc_tokens(text: str) -> set[str]:
    return set(_normalize_desc(text).split())


def _token_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _build_name_match_hints(items_a: list[dict[str, Any]], items_b: list[dict[str, Any]]) -> dict[str, Any]:
    b_norm_map: dict[str, list[int]] = {}
    b_desc_by_id: dict[int, str] = {}
    b_tokens_by_id: dict[int, set[str]] = {}

    for b in items_b:
        b_id = int(b["id"])
        b_desc = str(b.get("description") or "")
        b_norm = _normalize_desc(b_desc)
        b_desc_by_id[b_id] = b_desc
        b_tokens_by_id[b_id] = _desc_tokens(b_desc)
        if b_norm:
            b_norm_map.setdefault(b_norm, []).append(b_id)

    exact_name_candidates_by_a: dict[str, list[int]] = {}
    near_name_candidates_by_a: dict[str, list[dict[str, Any]]] = {}

    for a in items_a:
        a_id = int(a["id"])
        a_desc = str(a.get("description") or "")
        a_norm = _normalize_desc(a_desc)
        a_tokens = _desc_tokens(a_desc)

        exact_ids = list(b_norm_map.get(a_norm, []))
        if not exact_ids and a_tokens:
            for b_id, b_tokens in b_tokens_by_id.items():
                if a_tokens == b_tokens and len(a_tokens) >= 2:
                    exact_ids.append(int(b_id))
        if exact_ids:
            exact_name_candidates_by_a[str(a_id)] = exact_ids[:8]

        near: list[tuple[float, int]] = []
        for b_id, b_desc in b_desc_by_id.items():
            b_norm = _normalize_desc(b_desc)
            if not a_norm or not b_norm:
                continue
            seq = SequenceMatcher(None, a_norm, b_norm).ratio()
            jacc = _token_jaccard(a_tokens, b_tokens_by_id.get(b_id, set()))
            score = max(seq, jacc, (0.65 * seq) + (0.35 * jacc))
            if score >= 0.64:
                near.append((score, int(b_id)))
        near.sort(key=lambda x: x[0], reverse=True)
        if near:
            near_name_candidates_by_a[str(a_id)] = [
                {"item_b_id": b_id, "similarity": round(score, 3)}
                for score, b_id in near[:6]
            ]

    return {
        "exact_name_candidates_by_a": exact_name_candidates_by_a,
        "near_name_candidates_by_a": near_name_candidates_by_a,
    }


def _pct_diff(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return float("inf")
    denom = max(abs(b), 1e-9)
    return abs(a - b) / denom


def _metadata_alignment_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    score = 0.0
    weight = 0.0

    a_amount = a.get("amount")
    b_amount = b.get("amount")
    if a_amount is not None and b_amount is not None:
        score += max(0.0, 1.0 - min(1.0, _pct_diff(float(a_amount), float(b_amount))))
        weight += 1.6

    a_qty = a.get("quantity")
    b_qty = b.get("quantity")
    if a_qty is not None and b_qty is not None:
        score += max(0.0, 1.0 - min(1.0, _pct_diff(float(a_qty), float(b_qty))))
        weight += 1.0

    a_up = a.get("unit_price")
    b_up = b.get("unit_price")
    if a_up is not None and b_up is not None:
        score += max(0.0, 1.0 - min(1.0, _pct_diff(float(a_up), float(b_up))))
        weight += 1.0

    a_unit = str(a.get("unit") or "").strip().lower()
    b_unit = str(b.get("unit") or "").strip().lower()
    if a_unit and b_unit:
        score += 1.0 if a_unit == b_unit else 0.0
        weight += 0.8

    if weight <= 0.0:
        return 0.0
    return score / weight


def _enforce_exact_wording_matches(
    plan: MatchPlan,
    *,
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
) -> MatchPlan:
    hints = _build_name_match_hints(items_a, items_b)
    exact_by_a_raw = hints.get("exact_name_candidates_by_a", {})
    if not isinstance(exact_by_a_raw, dict) or not exact_by_a_raw:
        return plan

    a_by_id = {int(x["id"]): x for x in items_a}
    b_by_id = {int(x["id"]): x for x in items_b}
    best = _best_pairs_by_a(plan)

    # Keep exact matches already selected by model; build a baseline used set.
    used_b: set[int] = set()
    for a_id, pair in list(best.items()):
        exact_ids = [int(v) for v in (exact_by_a_raw.get(str(a_id)) or []) if int(v) in b_by_id]
        if pair.item_b_id is not None and int(pair.item_b_id) in exact_ids:
            used_b.add(int(pair.item_b_id))

    # Build exact-match assignment candidates with metadata tie-breaker.
    assignment_candidates: list[tuple[float, int, int]] = []
    for a_id_raw, b_ids_raw in exact_by_a_raw.items():
        a_id = int(a_id_raw)
        a_item = a_by_id.get(a_id)
        if a_item is None:
            continue
        for b_id_any in b_ids_raw:
            b_id = int(b_id_any)
            b_item = b_by_id.get(b_id)
            if b_item is None:
                continue
            meta_score = _metadata_alignment_score(a_item, b_item)
            assignment_candidates.append((meta_score, a_id, b_id))

    assignment_candidates.sort(key=lambda x: x[0], reverse=True)
    assigned_a: set[int] = set()
    assigned_b: set[int] = set(used_b)
    forced: dict[int, int] = {}
    for meta_score, a_id, b_id in assignment_candidates:
        if a_id in assigned_a or b_id in assigned_b:
            continue
        assigned_a.add(a_id)
        assigned_b.add(b_id)
        forced[a_id] = b_id

    if not forced:
        return plan

    for a_id, b_id in forced.items():
        a_item = a_by_id[a_id]
        b_item = b_by_id[b_id]
        meta_score = _metadata_alignment_score(a_item, b_item)
        current = best.get(a_id)
        rationale = (
            "exact-wording override: selected same-name candidate first; "
            f"metadata_alignment={meta_score:.2f} for downstream green/orange classification"
        )
        confidence = max(0.93, float(current.confidence) if current is not None else 0.0)
        best[a_id] = ProposedPair(
            item_a_id=a_id,
            item_b_id=b_id,
            scope_same=True,
            confidence=min(1.0, confidence),
            rationale=rationale,
            critical_blue=False,
        )

    # Ensure every A item still has one pair.
    pairs: list[ProposedPair] = []
    for a in items_a:
        a_id = int(a["id"])
        pair = best.get(a_id)
        if pair is not None:
            pairs.append(pair)
            continue
        pairs.append(
            ProposedPair(
                item_a_id=a_id,
                item_b_id=None,
                scope_same=False,
                confidence=0.0,
                rationale="no candidate selected",
                critical_blue=False,
            )
        )

    return MatchPlan(pairs=pairs)


def _to_brief_items(items: list[dict[str, Any]], max_len: int = 180) -> list[dict[str, Any]]:
    out = []
    for it in items:
        desc = (it["description"] or "").strip()
        if len(desc) > max_len:
            desc = desc[: max_len - 3] + "..."
        entry: dict[str, Any] = {
            "id": it["id"],
            "room": it["room"],
            "desc": desc,
            "amount": it.get("amount"),
        }
        if it.get("quantity") is not None:
            entry["qty"] = it["quantity"]
        if it.get("unit"):
            entry["unit"] = it["unit"]
        if it.get("unit_price") is not None:
            entry["unit_price"] = it["unit_price"]
        out.append(entry)
    return out


def _best_pairs_by_a(plan: MatchPlan) -> dict[int, ProposedPair]:
    best: dict[int, ProposedPair] = {}
    for p in plan.pairs:
        cur = best.get(p.item_a_id)
        if cur is None or p.confidence > cur.confidence:
            best[p.item_a_id] = p
    return best


def _normalize_model_list(models: Iterable[str]) -> list[str]:
    return normalize_model_list(models, _MATCH_ALIAS_MAP)


def _candidate_model_variants(model_name: str) -> list[str]:
    return expand_model_variants(model_name, _MATCH_ALIAS_MAP)


def _call_match_plan(
    *,
    model: str,
    room_a: str,
    room_b: str,
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    system_prompt: str,
    extra_user_fields: dict[str, Any] | None = None,
) -> MatchPlan:
    client = get_client()

    user: dict[str, Any] = {
        "room_a": room_a,
        "room_b": room_b,
        "items_a": _to_brief_items(items_a),
        "items_b": _to_brief_items(items_b),
        "instructions": (
            "Return a MatchPlan with one ProposedPair for each items_a entry. "
            "Process each item_a_id individually in order, and use exact_name_candidates_by_a / "
            "near_name_candidates_by_a before falling back to full-room comparison."
        ),
    }
    user.update(_build_name_match_hints(items_a, items_b))
    if extra_user_fields:
        user.update(extra_user_fields)

    message_body = json.dumps(user, ensure_ascii=True)
    last_error: Exception | None = None
    candidates = _candidate_model_variants(model)
    backup_models: list[str] = [settings.matching_first_pass_model]
    backup_models.extend(settings.matching_second_pass_model_list)
    backup_models.extend(
        [
            "openai/gpt-5",
            "openai/gpt-5.1",
            "anthropic/claude-3-5-sonnet-latest",
            "anthropic/claude-opus-4-5",
            "gemini/gemini-2.5-pro",
            "openai/gpt-4.1",
            "openai/gpt-4o-2024-08-06",
        ]
    )
    for backup_model in backup_models:
        for v in _candidate_model_variants(backup_model):
            if v not in candidates:
                candidates.append(v)

    for idx, model_candidate in enumerate(candidates):
        global _MATCH_ATTEMPTS, _MATCH_CALLS, _MATCH_MODEL_USAGE, _MATCH_FALLBACK_SUCCESSES
        _MATCH_ATTEMPTS += 1
        try:
            parsed = client.chat.completions.parse(
                model=model_candidate,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message_body},
                ],
                response_format=MatchPlan,
            ).choices[0].message.parsed
            _MATCH_CALLS += 1
            _MATCH_MODEL_USAGE[model_candidate] += 1
            if idx > 0:
                _MATCH_FALLBACK_SUCCESSES += 1
            return parsed
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No model candidate available for match-plan call.")


def propose_matches_for_room(
    room_a: str,
    room_b: str,
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> MatchPlan:
    chosen_model = model or settings.matching_first_pass_model
    plan = _call_match_plan(
        model=chosen_model,
        room_a=room_a,
        room_b=room_b,
        items_a=items_a,
        items_b=items_b,
        system_prompt=SYSTEM_PROMPT,
    )
    return _enforce_exact_wording_matches(plan, items_a=items_a, items_b=items_b)


def propose_matches_for_room_ensemble(
    room_a: str,
    room_b: str,
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    *,
    models: Iterable[str] | None = None,
    first_pass_by_a: dict[int, FirstPassDecision] | None = None,
    adjudicator_model: str | None = None,
) -> MatchPlan:
    """
    Lean second pass:
    1) Run one reviewer model over uncertain items using first-pass context.
    2) Fall back to first-pass only if reviewer call fails.
    """
    del adjudicator_model  # retained for call-site compatibility
    if not items_a:
        return MatchPlan(pairs=[])

    reviewer_models = _normalize_model_list(models or settings.matching_second_pass_model_list)
    reviewer_model = reviewer_models[0] if reviewer_models else settings.matching_first_pass_model

    review_context: list[dict[str, Any]] = []
    if first_pass_by_a:
        for item in items_a:
            a_id = int(item["id"])
            fp = first_pass_by_a.get(a_id)
            if fp is None:
                continue
            review_context.append(
                {
                    "item_a_id": a_id,
                    "first_pass_item_b_id": fp[0],
                    "first_pass_scope_same": bool(fp[1]),
                    "first_pass_confidence": float(fp[2]),
                    "first_pass_rationale": fp[3],
                }
            )

    try:
        reviewer_plan = _call_match_plan(
            model=reviewer_model,
            room_a=room_a,
            room_b=room_b,
            items_a=items_a,
            items_b=items_b,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            extra_user_fields={
                "first_pass_review_context": review_context,
                "reviewer_role": "single_second_pass_reviewer",
            },
        )
    except Exception:
        global _MATCH_REVIEWER_FALLBACKS
        _MATCH_REVIEWER_FALLBACKS += 1
        fallback = propose_matches_for_room(room_a, room_b, items_a, items_b)
        return MatchPlan(pairs=list(_best_pairs_by_a(fallback).values()))

    reviewer_plan = _enforce_exact_wording_matches(reviewer_plan, items_a=items_a, items_b=items_b)
    return MatchPlan(pairs=list(_best_pairs_by_a(reviewer_plan).values()))
