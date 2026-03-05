from __future__ import annotations

import base64
import re
from collections import Counter
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

TEXT_SYSTEM_PROMPT = """You extract ONLY estimate line items from insurance/contractor repair estimates.

Rules:
- Ignore cover pages, contact info, disclaimers, claim numbers, phone numbers, dates, policy text, and summary guides.
- Output ONLY real estimate rows (things that would be billed), not section/table headers.
- Assign each item to a room/area/section based on the nearest heading on the page.
- Extract all metadata columns when present: quantity, unit, unit_price (cost per unit), and total.
- If a field is missing or not shown for a row, set it to null.
- confidence should reflect how sure you are it is a real line item row.
"""

VISION_SYSTEM_PROMPT = """You extract ONLY estimate line items from an insurance/contractor repair estimate page shown as an image.

Rules:
- Read the table carefully. Identify each billable line item row.
- Ignore: cover pages, contact info, disclaimers, claim numbers, headers/footers, section headings, summary totals.
- For each real line item extract:
  - room: the room/area/section this item belongs to (from nearest section heading above it)
  - description: the line item description text
  - quantity: numeric quantity if shown, else null
  - unit: unit of measure (EA, SF, LF, HR, SY, etc) if shown, else null
  - unit_price: the price per unit if shown, else null
  - total: the total cost for this line item if shown, else null
  - confidence: 0.0-1.0, how sure you are this is a real billable line item
- Do NOT output section headers, subtotals, page totals, or non-billable rows.
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
    s = re.sub(r"^\d+\s*[\.\)]\s*", "", s)
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
        f"Extract line items from this page text:\n\n---BEGIN PAGE TEXT---\n{chunk}\n---END PAGE TEXT---"
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
            "text": f"Document: {doc}\nPage: {page_index}\n\nExtract all estimate line items from this page.",
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
                        return ExtractPageResult(doc=doc, page=page_index, items=page_items), {
                            "page": page_index,
                            "chunks_total": 1,
                            "chunks_with_items": 1,
                            "attempts": page_attempts,
                            "model_usage": dict(page_model_usage),
                            "vision_used": True,
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

    return ExtractPageResult(doc=doc, page=page_index, items=page_items), {
        "page": page_index,
        "chunks_total": len(chunks),
        "chunks_with_items": chunks_with_items,
        "attempts": page_attempts,
        "model_usage": dict(page_model_usage),
        "vision_used": vision_used,
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
                for model_name, count in (page_stats.get("model_usage") or {}).items():
                    model_usage[str(model_name)] += int(count)
            results = sorted(page_results, key=lambda r: int(r.page))

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
        "model_usage": dict(model_usage),
    }
