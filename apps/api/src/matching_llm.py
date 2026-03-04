from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from .llm_client import get_client
from .llm_match_schemas import MatchPlan, ProposedPair
from .settings import settings

SYSTEM_PROMPT = """You are matching line items between two repair estimates within the SAME room/area.

Goal:
- For each contractor item (Doc A), either match it to at most ONE insurance item (Doc B), or mark it unmatched.
- Matching is based on semantic scope similarity, not only wording or price.
- Do NOT match across rooms; assume lists provided are already room-scoped.

Rules:
- One-to-one: each B item can be used at most once.
- Provide one pair for every A item.
- If no plausible B match exists, return item_b_id=null.
- Do not overuse null. If a close semantic candidate exists in the same room, select it and explain metadata differences in rationale.
- IMPORTANT: Match using only the item name/description and the total amount.
- Ignore all other table columns and metadata (quantity, unit, unit price, tax, O&P, RESET/REMOVE/REPLACE subtotals, etc.).
"""

REVIEWER_SYSTEM_PROMPT = """You are a second-pass reviewer for uncertain line-item matches.

You will receive:
- room-scoped items from both documents
- first-pass decision and rationale per uncertain A item

Task:
- For each uncertain A item, evaluate the first-pass decision.
- If first-pass is correct, keep it.
- If incorrect, return the better B match or null.
- Keep null only when there is genuinely no same-room semantic candidate.
- IMPORTANT: Judge using only line-item name/description and total amount.
- Ignore quantity/unit/rate/tax/O&P/RESET/REMOVE/REPLACE and any other columns.
"""

_MODEL_ALIAS_MAP: dict[str, str] = {
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
        out.append(
            {
                "id": it["id"],
                "room": it["room"],
                "desc": desc,
                "amount": it["amount"],
            }
        )
    return out


def _best_pairs_by_a(plan: MatchPlan) -> dict[int, ProposedPair]:
    best: dict[int, ProposedPair] = {}
    for p in plan.pairs:
        cur = best.get(p.item_a_id)
        if cur is None or p.confidence > cur.confidence:
            best[p.item_a_id] = p
    return best


def _normalize_model_list(models: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in models:
        token = str(raw or "").strip()
        if not token:
            continue
        resolved = _MODEL_ALIAS_MAP.get(token.lower(), token)
        key = resolved.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _candidate_model_variants(model_name: str) -> list[str]:
    token = (model_name or "").strip()
    if not token:
        return []
    resolved = _MODEL_ALIAS_MAP.get(token.lower(), token)

    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        value = (name or "").strip()
        key = value.lower()
        if not value or key in seen:
            return
        seen.add(key)
        out.append(value)

    add(token)
    add(resolved)

    for base in list(out):
        if "/" in base:
            _, rhs = base.split("/", 1)
            add(rhs)
        else:
            lower = base.lower()
            if lower.startswith(("gpt-", "o1", "o3")):
                add(f"openai/{base}")
            if "claude" in lower:
                add(f"anthropic/{base}")
            if "gemini" in lower:
                add(f"gemini/{base}")
                add(f"google/{base}")
    return out


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
