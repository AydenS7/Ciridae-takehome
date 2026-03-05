from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from .llm_client import get_client
from .llm_match_schemas import MatchPlan, ProposedPair
from .llm_utils import expand_model_variants, normalize_model_list
from .settings import settings

SYSTEM_PROMPT = """You are matching line items between two repair estimates within the SAME room/area.

Goal:
- For each contractor item (Doc A), either match it to at most ONE insurance item (Doc B), or mark it unmatched.
- EVERY item in items_a MUST receive a ProposedPair. Do not skip any.
- Matching is based on the actual WORK being performed, not how it is worded.
- Do NOT match across rooms; assume lists provided are already room-scoped.

Rules:
- One-to-one: each B item can be used at most once.
- STRONGLY prefer matching over returning null. If any B item covers similar work, choose it.
- Focus on WHAT the work is, not how it is phrased. Examples of equivalent work:
    "demo and haul" = "debris removal" = "demolition and disposal"
    "replace carpet" = "carpet installation" = "carpet and pad replacement"
    "drywall repair" = "patch drywall" = "skim coat and repair"
    "paint walls" = "interior painting" = "apply finish coat"
    "water extraction" = "remove standing water" = "moisture mitigation"
- Do NOT return item_b_id=null just because descriptions use different wording. Match by work category.
- Return item_b_id=null ONLY when there is truly no B item for this type of work in this room.
- Metadata fields (amount, qty, unit, unit_price) inform your confidence and rationale but do NOT determine whether to match — that is handled by the classification system downstream.

scope_same field:
- Set scope_same=true whenever you select a B match (item_b_id is not null) AND both items represent the same general category of work (even if worded differently).
- Set scope_same=false only if you selected a B match but are genuinely uncertain whether the scopes truly overlap.
- Default to scope_same=true for any match you are reasonably confident about.

- For unmatched A items (item_b_id=null): set critical_blue=true if this item represents high-priority scope the insurer should cover — safety/electrical testing, permits, code compliance, environmental hazards (mold, asbestos, lead), engineering reports, specialized inspections, or items with significant liability implications.
"""

REVIEWER_SYSTEM_PROMPT = """You are a second-pass reviewer for uncertain line-item matches.

You will receive:
- room-scoped items from both documents
- first-pass decision and rationale per uncertain A item

Task:
- For each uncertain A item, evaluate the first-pass decision.
- If first-pass selected a B match: confirm it if reasonable, or return the better B if a clearly superior match exists.
- If first-pass returned null: AGGRESSIVELY look for any B item covering similar work. Override null whenever a plausible match exists.
- Keep null ONLY when there is truly no B item for this type of work in this room.
- Focus on WHAT the work IS, not how it is phrased — different wording does not mean different work.
- If amounts are in the same ballpark and the work category is similar (both are cleanup, both are flooring, both are painting, etc.), match them.
- scope_same: set true for any match where both items cover the same general type of work. Set false only when matched but genuinely uncertain about scope overlap.
- For items you confirm as null (unmatched): set critical_blue=true if the item represents high-priority scope — safety testing, permits, code compliance, environmental hazards, engineering reports, or items with significant liability implications.
"""

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
        "instructions": "Return a MatchPlan with one ProposedPair for each items_a entry.",
    }
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
    return _call_match_plan(
        model=chosen_model,
        room_a=room_a,
        room_b=room_b,
        items_a=items_a,
        items_b=items_b,
        system_prompt=SYSTEM_PROMPT,
    )


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

    return MatchPlan(pairs=list(_best_pairs_by_a(reviewer_plan).values()))
