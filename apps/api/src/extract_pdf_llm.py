"""LLM-powered line-item extraction pipeline with model fallback and telemetry."""

from __future__ import annotations

import base64
import re
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pdfplumber

from .llm_client import get_client
from .llm_schemas import ExtractPageResult, ExtractedItem
from .llm_utils import expand_model_variants
from .settings import settings

Doc = Literal["A", "B"]

# Models known to accept image_url content blocks.
_VISION_CAPABLE: frozenset[str] = frozenset({
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.0-pro",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-2024-08-06",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-3-5-sonnet-latest",
    "anthropic/claude-3-7-sonnet-latest",
})

TEXT_SYSTEM_PROMPT = """You extract ALL estimate line items from insurance/contractor repair estimates.

ITEM BOUNDARY RULE (most important):
- A new line item ALWAYS starts with a number followed by a period at the LEFT margin, e.g. "1.", "2.", "15."
- If a line does NOT start with "N.", it is a CONTINUATION of the previous item's description — NOT a new item.
- Concatenate continuation lines (text portion only) into the description of the item they belong to.
- Extract items in sequential order: 1, 2, 3, … up to the last number on the page. Do NOT skip any number.

INLINE METADATA RULE:
- The first line of an item often contains the description text followed by metadata columns inline, e.g.:
    "71. Door hinges (set of 3) and slab - 1.00 EA 0.00 29.94 0.00 5.98 35.92"
  Here "71. Door hinges (set of 3) and slab" is the description; "1.00 EA … 35.92" is metadata (qty/unit/prices).
- Continuation lines that follow (no leading number) add more description text. They may also contain inline metadata — extract the text portion and append it to the description. Example:
    "Detach & reset"                                          → append to description
    "Door hinges will need to be detached and reset …"       → append to description
  Full description becomes: "71. Door hinges (set of 3) and slab Detach & reset Door hinges will need to be detached and reset to allow replacement of jamb"

PROCESS — follow these steps in order:
1. Read the text top-to-bottom and identify every section/room heading (e.g. "Kitchen", "Master Bedroom", "Interior").
2. Use the "N." rule above to identify item boundaries. Any line without a leading "N." is part of the previous item.
3. Extract EVERY numbered item. After extracting, verify: is there a ProposedItem for every number from 1 (or the first number on this page) to the last number? If not, go back and add the missing ones.

Room assignment rules:
- Each item's room = the NEAREST section heading that appears ABOVE it in the text.
- When a new heading appears mid-page, ALL subsequent items belong to that new room — not the first heading on the page.
- A single page WILL often contain multiple rooms. Never assign every item on the page to the same room if multiple headings exist.

Per-item rules:
- description: the full line item description text including the leading "N." number (e.g. "1. Remove and replace drywall").
- Extract all metadata when present: quantity, unit, unit_price (cost per unit), and total.
- If a field is missing or not shown, set it to null.
- confidence = how sure you are this is a real billable row (0.0–1.0).
- Ignore: cover pages, contact info, disclaimers, claim numbers, phone numbers, dates, policy text, summary guides, section headers, subtotals, and page totals.
"""

VISION_SYSTEM_PROMPT = """You extract EVERY billable line item from an insurance/contractor repair estimate page shown as an image.

ITEM BOUNDARY RULE (most important):
- A new line item ALWAYS starts with a number followed by a period at the LEFT margin of the page, e.g. "1.", "2.", "15."
- If a row does NOT start with "N." at the left, it is a CONTINUATION of the previous item's description — NOT a new item. Append its text to the previous item's description.
- Extract items in sequential numerical order: 1, 2, 3, … to the last number visible. Do NOT skip any number.
- After extracting, count: is there one entry for every number from the first to the last on this page? If not, go back and add the missing ones.

INLINE METADATA RULE:
- An item's first line often has the description text followed by metadata columns on the same line, e.g.:
    "71. Door hinges (set of 3) and slab - 1.00 EA 0.00 29.94 0.00 5.98 35.92"
  Description = "71. Door hinges (set of 3) and slab"; metadata = qty 1.00, unit EA, total 35.92.
- Continuation rows (no leading number) that follow contribute MORE description text. Example:
    "Detach & reset"  and  "Door hinges will need to be detached and reset to allow replacement of jamb"
  are both continuations — append them. Full description: "71. Door hinges (set of 3) and slab Detach & reset Door hinges will need to be detached and reset to allow replacement of jamb"

PROCESS — follow these steps in order:
1. SCAN the full image top-to-bottom and note every section/room heading and its vertical position.
2. Use the "N." rule above to identify item boundaries. Process each numbered item in order.
3. After extracting all items, verify sequential completeness — no number may be skipped.

For each item extract:
  - room: the room/area/section this item belongs to — determined by the NEAREST heading ABOVE it on the page. Switch room whenever a new section heading appears above the items.
  - description: the full description including the leading "N." number (e.g. "3. Seal and prime walls"). If the item spans multiple lines, concatenate them.
  - quantity: numeric quantity if shown, else null
  - unit: unit of measure (EA, SF, LF, HR, SY, etc) if shown, else null
  - unit_price: the price per unit if shown, else null
  - total: the total cost for this line item if shown, else null
  - confidence: 0.0-1.0, how sure you are this is a real billable line item

CRITICAL rules:
- NEVER assign all items to the same room when multiple room headings are visible on the page.
- Track heading changes as you move DOWN the page; switch room whenever a new heading is passed.
- Do NOT output section headers, subtotals, page totals, or non-billable rows.
- Ignore: cover pages, contact info, disclaimers, claim numbers, headers/footers.
"""

_EXTRACT_ALIAS_MAP: dict[str, str] = {
    "openai": "openai/gpt-4.1",
    "chatgpt": "openai/gpt-4.1",
    "gpt5": "openai/gpt-5",
    "gpt-5": "openai/gpt-5",
    "anthropic": "anthropic/claude-opus-4-5",
    "claude": "anthropic/claude-opus-4-5",
    "gemini": "gemini/gemini-2.5-pro",
    "google": "gemini/gemini-2.5-pro",
}


def _resolve_model(model_name: str) -> str:
    token = (model_name or "").strip()
    if not token:
        return "gemini/gemini-2.5-pro"
    return _EXTRACT_ALIAS_MAP.get(token.lower(), token)


def _candidate_models(*names: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        for v in expand_model_variants(name, _EXTRACT_ALIAS_MAP):
            key = v.lower()
            if key and key not in seen:
                seen.add(key)
                out.append(v)
    return out


def _chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    lines = text.splitlines()
    chunks, cur = [], []
    cur_len = 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if cur_len + len(ln) + 1 > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _normalize_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (s or "").lower())


def _clean_description(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    return s.strip(" -:\t")


def _has_meaningful_description(text: str) -> bool:
    s = _clean_description(text)
    if len(s) < 6:
        return False
    toks = [t for t in _normalize_tokens(s) if len(t) > 2]
    if len(toks) < 2:
        return False
    alpha = sum(1 for ch in s if ch.isalpha())
    return alpha >= 4


_ITEM_LINE_RE = re.compile(r"^\s*(\d{1,4})\s*[\.\)]\s+")
# Matches JDR "Totals: Kitchen" / "Total - Master Bedroom" style end-of-section markers.
# Room name must start with a letter so we don't match "Total: $1,234.56".
_TOTALS_LINE_RE = re.compile(r"^\s*totals?\s*[:\-]?\s*([a-zA-Z].+)$", re.IGNORECASE)
_TABLE_HEADER_RE = re.compile(r"\b(description|qty|quantity|reset|remove|replace|tax|o&p|total|unit price|price)\b", re.IGNORECASE)
_NON_ROOM_HEADER_RE = re.compile(
    r"\b(subtotal|total|estimate|claim|policy|date|page|continued|deductible|overhead|profit)\b",
    re.IGNORECASE,
)


def _normalize_room_heading(line: str) -> str:
    s = re.sub(r"\s+", " ", (line or "").strip())
    s = re.sub(r"^\s*continued\s*-\s*", "", s, flags=re.IGNORECASE)
    s = s.strip(" -:\t")
    return s


def _is_room_heading_candidate(line: str) -> bool:
    s = _normalize_room_heading(line)
    if not s:
        return False
    if len(s) < 3 or len(s) > 72:
        return False
    if _ITEM_LINE_RE.match(s):
        return False
    if _TABLE_HEADER_RE.search(s):
        return False
    if _NON_ROOM_HEADER_RE.search(s):
        return False
    if "$" in s:
        return False
    words = re.findall(r"[a-z0-9']+", s.lower())
    if not words or len(words) > 7:
        return False
    # Heuristic: heading-like lines are mostly alphabetic and short.
    alpha_count = sum(1 for ch in s if ch.isalpha())
    if alpha_count < 3:
        return False
    digit_count = sum(1 for ch in s if ch.isdigit())
    if digit_count > 3:
        return False
    return True


def _item_number_to_room_from_page_text(page_text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not page_text:
        return mapping

    current_room: str | None = None
    for raw_line in page_text.splitlines():
        line = re.sub(r"\s+", " ", (raw_line or "").strip())
        if not line:
            continue

        if _is_room_heading_candidate(line):
            current_room = _normalize_room_heading(line)
            continue

        match = _ITEM_LINE_RE.match(line)
        if match and current_room:
            idx = int(match.group(1))
            mapping[idx] = current_room

    return mapping


def _item_number_from_description(description: str) -> int | None:
    m = _ITEM_LINE_RE.match((description or "").strip())
    if not m:
        return None
    return int(m.group(1))


def _reassign_rooms_by_heading_map(page_text: str, items: list[ExtractedItem]) -> tuple[list[ExtractedItem], int]:
    number_to_room = _item_number_to_room_from_page_text(page_text)
    if not number_to_room or not items:
        return items, 0

    out: list[ExtractedItem] = []
    changed = 0
    for it in items:
        idx = _item_number_from_description(it.description or "")
        target_room = number_to_room.get(idx) if idx is not None else None
        if target_room and target_room != it.room:
            out.append(
                ExtractedItem(
                    room=target_room,
                    description=it.description,
                    quantity=it.quantity,
                    unit=it.unit,
                    unit_price=it.unit_price,
                    total=it.total,
                    confidence=it.confidence,
                )
            )
            changed += 1
        else:
            out.append(it)
    return out, changed


def _parse_doc_item_rooms(page_texts: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """
    Document-level room assignment for JDR-style estimates where:
      - Items are numbered starting at 1 for each room section
      - Each section ENDS with a line like "Totals:  Kitchen" or "Totals - Master Bedroom"

    Reads all page texts sequentially. Accumulates (page, item_num) pairs in a pending
    list. When a "Totals: [Room Name]" line is found, assigns that room to all pending
    items (which may span multiple pages).

    Returns: [(page_num, item_num, room_name), ...] in document order.
    """
    pending: list[tuple[int, int]] = []
    assignments: list[tuple[int, int, str]] = []

    for page_num, text in sorted(page_texts, key=lambda x: x[0]):
        for raw_line in (text or "").splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip())
            if not line:
                continue

            totals_m = _TOTALS_LINE_RE.match(line)
            if totals_m:
                room_name = re.sub(r"\s+", " ", totals_m.group(1)).strip()
                # Strip trailing "continued" qualifier
                room_name = re.sub(r"\s*-?\s*continued\s*$", "", room_name, flags=re.IGNORECASE).strip()
                if room_name:
                    for p, n in pending:
                        assignments.append((p, n, room_name))
                    pending = []
                continue

            item_m = _ITEM_LINE_RE.match(line)
            if item_m:
                item_num = int(item_m.group(1))
                pending.append((page_num, item_num))

    # Any remaining pending items have no Totals — leave for LLM-assigned rooms
    return assignments


def _reassign_rooms_doc_level(
    doc_assignments: list[tuple[int, int, str]],
    all_results: list["ExtractPageResult"],
) -> tuple[list["ExtractPageResult"], int]:
    """
    After all LLM extractions, correct item rooms using document-level Totals-boundary map.
    Handles item number restarts across room sections on the same page.

    Returns (updated_results, total_changes).
    """
    if not doc_assignments:
        return all_results, 0

    # Build per-page ordered queue of (item_num, room) pairs in doc order
    page_queues: dict[int, deque[tuple[int, str]]] = defaultdict(deque)
    for page_num, item_num, room in doc_assignments:
        page_queues[page_num].append((item_num, room))

    updated: list["ExtractPageResult"] = []
    total_changes = 0

    for result in all_results:
        page_num = int(result.page)
        if page_num not in page_queues or not page_queues[page_num]:
            updated.append(result)
            continue

        # Work through items in item-number order (document order within page)
        remaining = list(page_queues[page_num])
        items_by_order = sorted(
            result.items,
            key=lambda it: _item_number_from_description(it.description or "") or 9999,
        )

        new_items: list[ExtractedItem] = []
        for it in items_by_order:
            item_num = _item_number_from_description(it.description or "")
            if item_num is None:
                new_items.append(it)
                continue
            # Find and consume the first matching entry in remaining
            matched_room: str | None = None
            for i, (q_num, q_room) in enumerate(remaining):
                if q_num == item_num:
                    matched_room = q_room
                    remaining.pop(i)
                    break
            if matched_room and matched_room != (it.room or "").strip():
                new_items.append(ExtractedItem(
                    room=matched_room,
                    description=it.description,
                    quantity=it.quantity,
                    unit=it.unit,
                    unit_price=it.unit_price,
                    total=it.total,
                    confidence=it.confidence,
                ))
                total_changes += 1
            else:
                new_items.append(it)

        updated.append(ExtractPageResult(doc=result.doc, page=result.page, items=new_items))

    return updated, total_changes


def _sanitize_items(items: list[ExtractedItem]) -> list[ExtractedItem]:
    cleaned: list[ExtractedItem] = []
    for it in items:
        room = re.sub(r"\s+", " ", (it.room or "").strip()) or "Unspecified"
        desc = _clean_description(it.description or "")
        if not _has_meaningful_description(desc):
            continue
        cleaned.append(
            ExtractedItem(
                room=room,
                description=desc,
                quantity=it.quantity,
                unit=it.unit,
                unit_price=it.unit_price,
                total=it.total,
                confidence=float(it.confidence),
            )
        )
    return cleaned


def _item_signature(item: ExtractedItem) -> tuple[str, str, float | None]:
    room_tokens = _normalize_tokens(item.room)[:4]
    desc = re.sub(r"^\d+\.\s*", "", (item.description or "").strip())
    desc_tokens = [t for t in _normalize_tokens(desc) if len(t) > 2][:9]
    amount = round(float(item.total), 2) if item.total is not None else None
    return (" ".join(room_tokens), " ".join(desc_tokens), amount)


def _render_page_to_b64(pdf_path: str, page_index: int, dpi: int = 150) -> str | None:
    """Render a PDF page (0-indexed) to a base64-encoded PNG via pymupdf."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        if page_index >= len(doc):
            return None
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        return base64.b64encode(pix.tobytes("png")).decode()
    except ImportError:
        return None
    except Exception:
        return None


def _extract_via_text(
    *,
    client,
    doc: Doc,
    page_index: int,
    chunk: str,
    candidates: list[str],
) -> tuple[list[ExtractedItem], str | None, int]:
    user = (
        f"Document: {doc}\nPage: {page_index}\n\n"
        f"Extract ALL line items from this page text. Remember: a new item always starts with 'N.' at the left margin. "
        f"Lines without a leading number are continuations of the previous item — do not treat them as new items. "
        f"Go sequentially from the first number to the last; every number must appear exactly once in your output.\n\n"
        f"---BEGIN PAGE TEXT---\n{chunk}\n---END PAGE TEXT---"
    )
    last_error: Exception | None = None
    for idx, candidate in enumerate(candidates):
        try:
            parsed = client.chat.completions.parse(
                model=candidate,
                messages=[
                    {"role": "system", "content": TEXT_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format=ExtractPageResult,
            ).choices[0].message.parsed
            return _sanitize_items([it for it in parsed.items if float(it.confidence) >= 0.0]), candidate, idx + 1
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return [], None, len(candidates)


def _extract_via_vision(
    *,
    client,
    b64_image: str,
    doc: Doc,
    page_index: int,
    candidates: list[str],
) -> tuple[list[ExtractedItem], str | None, int]:
    user_content = [
        {
            "type": "text",
            "text": (
                f"Document: {doc}\nPage: {page_index}\n\n"
                "Scan top-to-bottom: identify every room/section heading and its position on the page. "
                "A new item ALWAYS starts with 'N.' at the LEFT margin (e.g. '1.', '12.'). "
                "Any row that does NOT start with a number-period is a continuation of the previous item — do not split it out as a new item. "
                "Extract items in sequential order; every number from first to last must appear exactly once. "
                "Switch room whenever a new section heading appears above the items."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
        },
    ]
    last_error: Exception | None = None
    for idx, candidate in enumerate(candidates):
        try:
            parsed = client.chat.completions.parse(
                model=candidate,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=ExtractPageResult,
            ).choices[0].message.parsed
            return _sanitize_items([it for it in parsed.items if float(it.confidence) >= 0.0]), candidate, idx + 1
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return [], None, len(candidates)


def _merge_dual_model_items(primary_items: list[ExtractedItem], secondary_items: list[ExtractedItem]) -> list[ExtractedItem]:
    primary_by_sig: dict[tuple, ExtractedItem] = {}
    secondary_by_sig: dict[tuple, ExtractedItem] = {}

    for it in primary_items:
        sig = _item_signature(it)
        cur = primary_by_sig.get(sig)
        if cur is None or float(it.confidence) > float(cur.confidence):
            primary_by_sig[sig] = it

    for it in secondary_items:
        sig = _item_signature(it)
        cur = secondary_by_sig.get(sig)
        if cur is None or float(it.confidence) > float(cur.confidence):
            secondary_by_sig[sig] = it

    merged: list[ExtractedItem] = []
    min_conf = min(float(settings.extract_min_confidence_primary), float(settings.extract_min_confidence_secondary))
    for sig in set(primary_by_sig.keys()) | set(secondary_by_sig.keys()):
        p = primary_by_sig.get(sig)
        s = secondary_by_sig.get(sig)
        if p is not None and s is not None:
            chosen = p if float(p.confidence) >= float(s.confidence) else s
            merged_conf = min(1.0, (float(p.confidence) + float(s.confidence)) / 2.0 + 0.08)
            if merged_conf >= min_conf:
                merged.append(
                    ExtractedItem(
                        room=chosen.room,
                        description=chosen.description,
                        quantity=chosen.quantity,
                        unit=chosen.unit,
                        unit_price=chosen.unit_price,
                        total=chosen.total,
                        confidence=merged_conf,
                    )
                )
        elif p is not None and float(p.confidence) >= float(settings.extract_min_confidence_primary):
            merged.append(p)
        elif s is not None and float(s.confidence) >= float(settings.extract_secondary_only_min_confidence):
            merged.append(s)

    return _sanitize_items(merged)


def _extract_page(
    *,
    doc: Doc,
    page_index: int,
    text: str,
    pdf_path: str,
    primary_model: str,
    secondary_model: str,
    secondary_enabled: bool,
    chunk_max_chars: int,
    vision_enabled: bool,
    vision_candidates: list[str],
) -> tuple[ExtractPageResult, dict]:
    client = get_client()
    text_candidates = _candidate_models(
        primary_model, secondary_model,
        settings.extract_primary_model, settings.extract_secondary_model,
        "openai/gpt-4.1", "openai/gpt-4o-2024-08-06",
    )
    page_items: list[ExtractedItem] = []
    page_model_usage: Counter[str] = Counter()
    page_attempts = 0
    chunks_with_items = 0
    vision_used = False
    room_reassignments = 0
    min_conf_primary = float(settings.extract_min_confidence_primary)

    # Vision extraction — primary path when enabled
    if vision_enabled and vision_candidates:
        b64 = _render_page_to_b64(pdf_path, page_index - 1)  # pdfplumber is 1-indexed; pymupdf is 0-indexed
        if b64:
            try:
                v_items, v_model, v_attempts = _extract_via_vision(
                    client=client,
                    b64_image=b64,
                    doc=doc,
                    page_index=page_index,
                    candidates=vision_candidates,
                )
                page_attempts += v_attempts
                if v_model:
                    page_model_usage[v_model] += 1
                    vision_used = True
                if v_items:
                    page_items = [it for it in v_items if float(it.confidence) >= min_conf_primary]
                    if page_items:
                        page_items, changed = _reassign_rooms_by_heading_map(text, page_items)
                        room_reassignments += changed
                        return ExtractPageResult(doc=doc, page=page_index, items=page_items), {
                            "page": page_index,
                            "chunks_total": 1,
                            "chunks_with_items": 1,
                            "attempts": page_attempts,
                            "model_usage": dict(page_model_usage),
                            "vision_used": True,
                            "room_reassignments": room_reassignments,
                        }
            except Exception:
                pass  # fall through to text extraction

    # Text extraction — fallback or sole method when vision is disabled
    chunks = _chunk_text(text, max_chars=max(6000, chunk_max_chars))
    for chunk in chunks:
        primary_items: list[ExtractedItem] = []
        secondary_items: list[ExtractedItem] = []

        try:
            primary_items, p_model, p_attempts = _extract_via_text(
                client=client, doc=doc, page_index=page_index, chunk=chunk, candidates=text_candidates,
            )
            page_attempts += p_attempts
            if p_model:
                page_model_usage[p_model] += 1
        except Exception:
            primary_items = []

        if secondary_enabled:
            secondary_candidates = _candidate_models(secondary_model, primary_model)
            try:
                secondary_items, s_model, s_attempts = _extract_via_text(
                    client=client, doc=doc, page_index=page_index, chunk=chunk, candidates=secondary_candidates,
                )
                page_attempts += s_attempts
                if s_model:
                    page_model_usage[s_model] += 1
            except Exception:
                secondary_items = []

        before_count = len(page_items)
        if primary_items and secondary_items:
            page_items.extend(_merge_dual_model_items(primary_items, secondary_items))
        elif primary_items:
            page_items.extend([it for it in _sanitize_items(primary_items) if float(it.confidence) >= min_conf_primary])
        elif secondary_items:
            page_items.extend(
                [it for it in _sanitize_items(secondary_items) if float(it.confidence) >= float(settings.extract_secondary_only_min_confidence)]
            )
        if len(page_items) > before_count:
            chunks_with_items += 1

    page_items, changed = _reassign_rooms_by_heading_map(text, page_items)
    room_reassignments += changed
    return ExtractPageResult(doc=doc, page=page_index, items=page_items), {
        "page": page_index,
        "chunks_total": len(chunks),
        "chunks_with_items": chunks_with_items,
        "attempts": page_attempts,
        "model_usage": dict(page_model_usage),
        "vision_used": vision_used,
        "room_reassignments": room_reassignments,
    }


def extract_pdf_via_llm(pdf_path: str, doc: Doc) -> tuple[list[ExtractPageResult], dict]:
    primary_model = _resolve_model(settings.extract_primary_model)
    secondary_model = _resolve_model(settings.extract_secondary_model)
    secondary_enabled = bool(settings.extract_enable_secondary)
    vision_enabled = bool(settings.extract_enable_vision)
    chunk_max_chars = int(settings.extract_chunk_max_chars)
    max_workers = max(1, int(settings.extract_max_workers))
    model_usage: Counter[str] = Counter()
    pages_seen = pages_processed = chunks_total = chunks_with_items = model_attempts = vision_pages = 0
    room_reassignments_total = 0

    # Build ordered list of vision-capable model candidates
    vision_model = _resolve_model(settings.extract_vision_model)
    vision_candidates: list[str] = []
    seen_v: set[str] = set()
    for name in (vision_model, primary_model, "gemini/gemini-2.5-pro", "openai/gpt-4.1", "openai/gpt-4o-2024-08-06"):
        for v in expand_model_variants(name, _EXTRACT_ALIAS_MAP):
            key = v.lower()
            is_vision = any(v.lower() == vm or v.lower().startswith(vm + "/") for vm in _VISION_CAPABLE) or key in {w.lower() for w in _VISION_CAPABLE}
            if key and key not in seen_v and is_vision:
                seen_v.add(key)
                vision_candidates.append(v)

    candidate_pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            pages_seen += 1
            text = (page.extract_text(x_tolerance=2, y_tolerance=2) or "").strip()
            # Keep image-only pages when vision extraction is enabled.
            if not text and not vision_enabled:
                continue
            pages_processed += 1
            candidate_pages.append((page_index, text))

    # Document-level room map: "Totals: [Room]" boundaries tell us which room each item
    # belongs to. This is built from raw page text BEFORE LLM extraction so it works even
    # when items and their "Totals:" line are on different pages.
    doc_item_rooms = _parse_doc_item_rooms(candidate_pages)

    results: list[ExtractPageResult] = []
    if candidate_pages:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _extract_page,
                    doc=doc,
                    page_index=page_index,
                    text=text,
                    pdf_path=pdf_path,
                    primary_model=primary_model,
                    secondary_model=secondary_model,
                    secondary_enabled=secondary_enabled,
                    chunk_max_chars=chunk_max_chars,
                    vision_enabled=vision_enabled,
                    vision_candidates=vision_candidates,
                ): page_index
                for page_index, text in candidate_pages
            }
            page_results: list[ExtractPageResult] = []
            for fut in as_completed(futures):
                page_result, page_stats = fut.result()
                page_results.append(page_result)
                chunks_total += int(page_stats.get("chunks_total", 0))
                chunks_with_items += int(page_stats.get("chunks_with_items", 0))
                model_attempts += int(page_stats.get("attempts", 0))
                if page_stats.get("vision_used"):
                    vision_pages += 1
                room_reassignments_total += int(page_stats.get("room_reassignments", 0))
                for model_name, count in (page_stats.get("model_usage") or {}).items():
                    model_usage[str(model_name)] += int(count)
            page_results = sorted(page_results, key=lambda r: int(r.page))

        # Apply document-level Totals-boundary room correction AFTER all pages extracted.
        # This overrides LLM-assigned rooms with the authoritative "Totals: Room" markers.
        if doc_item_rooms:
            page_results, doc_level_changes = _reassign_rooms_doc_level(doc_item_rooms, page_results)
            room_reassignments_total += doc_level_changes

        results = page_results

    return results, {
        "doc": doc,
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "secondary_enabled": secondary_enabled,
        "vision_enabled": vision_enabled,
        "vision_pages": vision_pages,
        "max_workers": max_workers,
        "chunk_max_chars": chunk_max_chars,
        "pages_seen": pages_seen,
        "pages_processed": pages_processed,
        "chunks_total": chunks_total,
        "chunks_with_items": chunks_with_items,
        "model_attempts": model_attempts,
        "room_reassignments": room_reassignments_total,
        "model_usage": dict(model_usage),
    }
