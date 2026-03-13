"""PDF annotation renderer for reconciliation highlights and summary notes."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Highlight, Popup, Text
from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject


def _money(x):
    if x is None:
        return "-"
    return f"${x:,.2f}"


def _cut(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _pct_diff(a, b) -> float | None:
    if a is None or b is None:
        return None
    denom = max(abs(b), 1e-9)
    return (abs(a - b) / denom) * 100.0


def _normalized_tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", (s or "").lower()) if t]


def _description_candidates(description: str) -> list[str]:
    raw = re.sub(r"\s+", " ", (description or "").strip())
    if not raw:
        return []

    # If the description starts with "N." extract number-anchored candidates as primary locators.
    # The item number is a reliable unique anchor in the source PDF.
    num_anchor_candidates: list[str] = []
    num_match = re.match(r"^(\d+)\.\s*(.+)", raw)
    if num_match:
        num = num_match.group(1)
        rest = num_match.group(2).strip()
        rest_words = rest.split()
        # "N. first two words" — tight, unique anchor
        num_first2 = f"{num}. {' '.join(rest_words[:2])}" if len(rest_words) >= 2 else f"{num}. {rest}"
        # "N. first four words" — slightly longer anchor
        num_first4 = f"{num}. {' '.join(rest_words[:4])}" if len(rest_words) >= 4 else None
        num_anchor_candidates.append(num_first2)
        if num_first4:
            num_anchor_candidates.append(num_first4)

    no_num = re.sub(r"^\d+\.\s*", "", raw)
    no_qty = re.sub(r"\b\d+(?:\.\d+)?\s*(?:EA|LF|SF|SQ|SY|HR|MO|WK|DAY)\b", "", no_num, flags=re.IGNORECASE)
    no_money = re.sub(r"\$?\d[\d,]*(?:\.\d{2})?", "", no_qty)
    compact = re.sub(r"\s+", " ", no_money).strip(" -:")

    words = compact.split()
    first4 = " ".join(words[:4]) if len(words) >= 4 else compact
    first6 = " ".join(words[:6]) if len(words) >= 6 else compact
    first8 = " ".join(words[:8]) if len(words) >= 8 else compact

    # Number-anchored candidates come first so the item number is the primary search target.
    candidates = num_anchor_candidates + [raw, no_num, compact, first8, first6, first4]
    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        if len(c) < 4:
            continue
        low = c.lower()
        if low in seen:
            continue
        seen.add(low)
        ordered.append(c)
    return ordered


def _sort_matches(matches: list[dict], page_width: float) -> list[dict]:
    if not matches:
        return []
    left_side = [m for m in matches if float(m.get("x0", 0)) <= page_width * 0.55]
    pool = left_side if left_side else matches
    return sorted(pool, key=lambda m: (float(m.get("top", 0)), float(m.get("x0", 0))))


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-9)
    area_b = max((bx1 - bx0) * (by1 - by0), 1e-9)
    return inter / min(area_a, area_b)


def _is_used_rect(rect: tuple[float, float, float, float], used_rects: list[tuple[float, float, float, float]]) -> bool:
    x0, y0, x1, y1 = rect
    cy = (y0 + y1) / 2.0
    for u in used_rects:
        if _overlap_ratio(rect, u) >= 0.85:
            return True
        ux0, uy0, ux1, uy1 = u
        ucy = (uy0 + uy1) / 2.0
        # Treat close same-line regions as already used so we pick the next row occurrence.
        if abs(cy - ucy) <= 2.8 and abs(x0 - ux0) <= 180:
            return True
    return False


def _rect_from_hit(page_height: float, page_width: float, hit: dict) -> tuple[float, float, float, float] | None:
    x0 = max(float(hit["x0"]) - 1.5, 0.0)
    x1 = min(float(hit["x1"]) + 1.5, page_width)
    y_top = max(float(hit["top"]) - 0.8, 0.0)
    y_bottom = min(float(hit["bottom"]) + 0.8, page_height)
    y1 = page_height - y_top
    y0 = page_height - y_bottom
    if y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _find_match_rect(
    page: pdfplumber.page.Page,
    description: str,
    used_rects: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    page_width = float(page.width)
    page_height = float(page.height)
    first_candidate_rect: tuple[float, float, float, float] | None = None

    for candidate in _description_candidates(description):
        hits = page.search(
            candidate,
            regex=False,
            case=False,
            return_chars=False,
            return_groups=False,
        )
        for hit in _sort_matches(hits, page_width):
            rect = _rect_from_hit(page_height, page_width, hit)
            if rect is None:
                continue
            if first_candidate_rect is None:
                first_candidate_rect = rect
            if not _is_used_rect(rect, used_rects):
                return rect

    tokens = [t for t in _normalized_tokens(description) if len(t) >= 3][:4]
    if len(tokens) >= 3:
        pattern = r"\b" + r"\W+".join(re.escape(t) for t in tokens) + r"\b"
        hits = page.search(
            pattern,
            regex=True,
            case=False,
            return_chars=False,
            return_groups=False,
        )
        for hit in _sort_matches(hits, page_width):
            rect = _rect_from_hit(page_height, page_width, hit)
            if rect is None:
                continue
            if first_candidate_rect is None:
                first_candidate_rect = rect
            if not _is_used_rect(rect, used_rects):
                return rect

    # Final fallback: search by bare item number ("26. ").
    # Item numbers are unique sequential anchors on every estimate page.
    num_m = re.match(r"^\s*(\d+)\s*[\.\)]\s*", description or "")
    if num_m:
        num_str = num_m.group(1)
        for num_pattern in (f"{num_str}. ", f"{num_str}."):
            hits = page.search(
                num_pattern,
                regex=False,
                case=False,
                return_chars=False,
                return_groups=False,
            )
            for hit in _sort_matches(hits, page_width):
                rect = _rect_from_hit(page_height, page_width, hit)
                if rect is None:
                    continue
                if first_candidate_rect is None:
                    first_candidate_rect = rect
                if not _is_used_rect(rect, used_rects):
                    return rect

    # If all candidates were already used, reuse the best one rather than dropping the comment.
    return first_candidate_rect


def _find_room_anchor_rect(
    page: pdfplumber.page.Page,
    room_name: str,
) -> tuple[float, float, float, float] | None:
    room = re.sub(r"\s+", " ", (room_name or "").strip())
    if not room:
        return None
    page_width = float(page.width)
    page_height = float(page.height)
    candidates = [room]
    toks = _normalized_tokens(room)
    if len(toks) >= 2:
        candidates.append(" ".join(toks[:2]))
    if toks:
        candidates.append(toks[0])
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        hits = page.search(
            candidate,
            regex=False,
            case=False,
            return_chars=False,
            return_groups=False,
        )
        sorted_hits = _sort_matches(hits, page_width)
        if not sorted_hits:
            continue
        for hit in sorted_hits:
            rect = _rect_from_hit(page_height, page_width, hit)
            if rect is not None:
                return rect
    return None


def _quad_points_for_rect(rect: tuple[float, float, float, float]) -> ArrayObject:
    x0, y0, x1, y1 = rect
    # PDF quad points: top-left, top-right, bottom-left, bottom-right.
    return ArrayObject(
        [
            FloatObject(x0),
            FloatObject(y1),
            FloatObject(x1),
            FloatObject(y1),
            FloatObject(x0),
            FloatObject(y0),
            FloatObject(x1),
            FloatObject(y0),
        ]
    )


def _item_number_prefix(desc: str) -> str:
    m = re.match(r"^\s*(\d+)\s*[\.\)]\s*", (desc or ""))
    return f"[#{m.group(1)}] " if m else ""


def _comment_for_row(row: dict) -> str:
    status = (row.get("status") or "").lower()
    item_prefix = _item_number_prefix(row.get("a_desc", ""))
    room_a = _cut(row.get("room_a", ""), 50)
    room_b = _cut(row.get("room_b", ""), 50)
    a_desc = _cut(row.get("a_desc", ""), 140)
    b_desc = _cut(row.get("b_desc", ""), 140)
    rationale = _cut(row.get("rationale", ""), 220)
    a_amt = row.get("a_amt")
    b_amt = row.get("b_amt")
    diff_pct = _pct_diff(a_amt, b_amt)
    diff_str = f"{diff_pct:.1f}%" if diff_pct is not None else "n/a"

    if status == "blue":
        critical_blue = bool(row.get("critical_blue"))
        a_desc_lower = (a_desc or "").lower()
        if critical_blue and ("megohmmeter" in a_desc_lower or ("electrical" in a_desc_lower and "test" in a_desc_lower)):
            return (
                f"{item_prefix}The adjuster's estimate does not include any line item for electrical testing such as a "
                "Megohmmeter check; it appears this work was excluded from the adjuster's scope."
            )
        if critical_blue:
            return (
                f"{item_prefix}Room: {room_a}. "
                f'High-priority JDR-only scope item "{a_desc}" at {_money(a_amt)} was not found in the insurance estimate. '
                "Review for potential scope omission."
            )
        return (
            f"{item_prefix}Room: {room_a}. "
            f'The contractor estimate includes "{a_desc}" at {_money(a_amt)}, '
            "but no comparable item was found in the insurance estimate."
        )

    if status == "orange":
        a_qty = row.get("a_qty")
        b_qty = row.get("b_qty")
        a_up = row.get("a_unit_price")
        b_up = row.get("b_unit_price")
        a_unit = (row.get("a_unit") or "").strip()
        b_unit = (row.get("b_unit") or "").strip()
        meta_parts: list[str] = []
        if diff_pct is not None:
            meta_parts.append(f"total diff {diff_str}")
        if a_qty is not None and b_qty is not None and abs(a_qty - b_qty) > 1e-6:
            meta_parts.append(f"qty: {a_qty} vs {b_qty}")
        if a_up is not None and b_up is not None and abs(a_up - b_up) > 1e-6:
            meta_parts.append(f"unit price: {_money(a_up)} vs {_money(b_up)}")
        if a_unit and b_unit and a_unit.upper() != b_unit.upper():
            meta_parts.append(f"unit: {a_unit} vs {b_unit}")
        meta_str = ("; ".join(meta_parts)) if meta_parts else f"difference: {diff_str}"
        core = (
            f"{item_prefix}Rooms: {room_a} vs {room_b}. "
            f'Contractor lists "{a_desc}" at {_money(a_amt)} while insurance lists '
            f'"{b_desc}" at {_money(b_amt)} ({meta_str}).'
        )
        if rationale:
            core = f"{core} {rationale}"
        return core

    if status == "green":
        a_qty = row.get("a_qty")
        b_qty = row.get("b_qty")
        a_up = row.get("a_unit_price")
        b_up = row.get("b_unit_price")
        meta_parts: list[str] = []
        if diff_pct is not None:
            meta_parts.append(f"total diff {diff_str}")
        if a_qty is not None or b_qty is not None:
            meta_parts.append(f"qty: {a_qty} vs {b_qty}")
        if a_up is not None or b_up is not None:
            meta_parts.append(f"unit price: {_money(a_up)} vs {_money(b_up)}")
        meta_str = ("; ".join(meta_parts)) if meta_parts else "within tolerance"
        return (
            f"{item_prefix}Rooms: {room_a} vs {room_b}. "
            f'Items align: contractor "{a_desc}" at {_money(a_amt)} and insurance "{b_desc}" at {_money(b_amt)} '
            f"({meta_str})."
        )

    if status == "nugget":
        return (
            f"Insurance-only nugget for mapped room {room_a}. "
            f'Insurance includes "{b_desc}" at {_money(b_amt)} with no matched JDR line item.'
        )

    return (
        f"{item_prefix}Rooms: {room_a} vs {room_b}. "
        f'Contractor "{a_desc}" {_money(a_amt)} vs insurance "{b_desc}" {_money(b_amt)}.'
    )


def _status_highlight_color(status: str) -> str:
    status = (status or "").lower()
    # Hex RGB for Adobe highlight rendering.
    if status == "green":
        return "86efac"  # light green
    if status == "orange":
        return "fdba74"  # light orange
    if status == "blue":
        return "93c5fd"  # light blue
    if status == "nugget":
        return "fde68a"  # warm yellow
    return "f3e5ab"  # fallback amber


def _status_note_color_array(status: str) -> ArrayObject:
    status = (status or "").lower()
    if status == "green":
        rgb = (0x22, 0xC5, 0x5E)
    elif status == "orange":
        rgb = (0xF9, 0x73, 0x16)
    elif status == "blue":
        rgb = (0x3B, 0x82, 0xF6)
    elif status == "nugget":
        rgb = (0xE1, 0xA6, 0x00)
    else:
        rgb = (0xF3, 0xE5, 0xAB)
    return ArrayObject([FloatObject(rgb[0] / 255), FloatObject(rgb[1] / 255), FloatObject(rgb[2] / 255)])


# --- DEBUG: room-based highlight colors ---
_ROOM_COLOR_PALETTE = [
    "93c5fd",  # blue
    "86efac",  # green
    "fde68a",  # yellow
    "fdba74",  # orange
    "c4b5fd",  # purple
    "f9a8d4",  # pink
    "6ee7b7",  # teal
    "fca5a5",  # red
    "d9f99d",  # lime
    "a5f3fc",  # cyan
]

_ROOM_KEYWORD_COLORS: dict[str, str] = {
    "bathroom": "93c5fd",   # blue
    "bath":     "93c5fd",
    "bedroom":  "86efac",   # green
    "bed":      "86efac",
    "hallway":  "fde68a",   # yellow
    "hall":     "fde68a",
    "kitchen":  "fdba74",   # orange
    "living":   "c4b5fd",   # purple
    "dining":   "f9a8d4",   # pink
    "garage":   "d1d5db",   # gray
    "basement": "d4a574",   # brown
    "office":   "6ee7b7",   # teal
    "laundry":  "ddd6fe",   # lavender
    "exterior": "fca5a5",   # red
    "stair":    "d9f99d",   # lime
    "attic":    "a5f3fc",   # cyan
}

_room_color_cache: dict[str, str] = {}
_room_color_counter: list[int] = [0]


def _room_highlight_color(room_name: str) -> str:
    """Return a stable hex highlight color for a room name (debug mode)."""
    key = (room_name or "").strip().lower()
    if key in _room_color_cache:
        return _room_color_cache[key]
    for keyword, color in _ROOM_KEYWORD_COLORS.items():
        if keyword in key:
            _room_color_cache[key] = color
            return color
    # Assign next palette color for unknown rooms
    idx = _room_color_counter[0] % len(_ROOM_COLOR_PALETTE)
    color = _ROOM_COLOR_PALETTE[idx]
    _room_color_counter[0] += 1
    _room_color_cache[key] = color
    return color


def _nugget_line(row: dict) -> str:
    room_b = _cut(row.get("room_b", ""), 36)
    b_desc = _cut(row.get("b_desc", ""), 90)
    b_amt = row.get("b_amt")
    return f"{room_b}: {b_desc} ({_money(b_amt)})"


def _note_rect_near_line(
    rect: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    *,
    note_size: float = 14.0,
    gap: float = 3.0,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    y_mid = (y0 + y1) / 2.0
    note_y0 = max(2.0, min(y_mid - note_size / 2.0, page_height - note_size - 2.0))
    note_y1 = note_y0 + note_size

    # Prefer icon to the right of the highlighted text, otherwise place it to the left.
    note_x0 = x1 + gap
    note_x1 = note_x0 + note_size
    if note_x1 > page_width - 2.0:
        note_x1 = max(2.0 + note_size, x0 - gap)
        note_x0 = note_x1 - note_size

    if note_x0 < 2.0:
        note_x0 = 2.0
        note_x1 = note_x0 + note_size

    return (note_x0, note_y0, note_x1, note_y1)


def _fallback_note_rect(page_width: float, page_height: float, index: int) -> tuple[float, float, float, float]:
    note_size = 14.0
    x0 = max(2.0, page_width - 20.0)
    x1 = min(page_width - 2.0, x0 + note_size)
    y1 = max(16.0, page_height - 24.0 - (index * (note_size + 2.0)))
    y0 = max(2.0, y1 - note_size)
    return (x0, y0, x1, y1)


def render_report_pdf(
    out_path: str,
    source_pdf_path: str,
    run_id: str,
    rows: list[dict],
    summary: dict,
) -> tuple[str, dict]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    src = Path(source_pdf_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    reader = PdfReader(str(src))
    writer = PdfWriter()

    # Determine how many pages to include in the output.
    # Stop at whichever comes first:
    #   1. A page containing "Summary for Dwelling" — stop AFTER that page (it's a summary,
    #      everything following is photos/appendices with no line items).
    #   2. An image-dominant page — stop BEFORE it (few words + images cover >30% of area).
    def _is_image_page(pg) -> bool:
        words = pg.extract_words()
        if len(words) >= 20:
            return False
        page_area = float(pg.width or 1) * float(pg.height or 1)
        image_area = sum(
            abs((img.get("x1", 0) - img.get("x0", 0)) * (img.get("y1", 0) - img.get("y0", 0)))
            for img in (pg.images or [])
        )
        return (image_area / page_area) > 0.30

    last_page_to_include = len(reader.pages)  # default: all pages (exclusive upper bound)
    with pdfplumber.open(str(src)) as _probe:
        for i, pg in enumerate(_probe.pages):
            text = (pg.extract_text() or "").lower()
            if "summary for dwelling" in text:
                last_page_to_include = i + 1  # include this page, stop after
                break
            if _is_image_page(pg):
                last_page_to_include = i  # exclude this page and everything after
                break

    for i, page in enumerate(reader.pages):
        if i >= last_page_to_include:
            break
        writer.add_page(page)

    if len(writer.pages) == 0:
        with out.open("wb") as f:
            writer.write(f)
        return str(out), {
            "line_items_targeted": 0,
            "highlights_added": 0,
            "inline_notes_added": 0,
            "unlocated_notes_added": 0,
            "anchored_room_notes_added": 0,
        }

    def _item_num(desc: str) -> int:
        m = re.match(r"^\s*(\d+)\s*[\.\)]\s*", (desc or ""))
        return int(m.group(1)) if m else 9999

    status_rank = {"orange": 0, "blue": 1, "green": 2, "nugget": 3}
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int(r.get("a_page") or 10_000),
            _item_num(r.get("a_desc", "")),
            status_rank.get((r.get("status") or "").lower(), 99),
            (r.get("room_a") or "").lower(),
        ),
    )

    used_rects_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    fallback_note_count_by_page: dict[int, int] = {}
    seen_a_item_ids: set[int] = set()
    render_stats = {
        "line_items_targeted": 0,
        "highlights_added": 0,
        "inline_notes_added": 0,
        "unlocated_notes_added": 0,
        "anchored_room_notes_added": 0,
        "nugget_summary_notes": 0,
    }
    with pdfplumber.open(str(src)) as plumber_pdf:
        for row in rows_sorted:
            status = (row.get("status") or "").lower()
            if status not in {"green", "orange", "blue"}:
                continue

            a_item_id = row.get("a_item_id")
            if isinstance(a_item_id, int):
                if a_item_id in seen_a_item_ids:
                    continue
                seen_a_item_ids.add(a_item_id)

            page_number = row.get("a_page")
            if not isinstance(page_number, int):
                continue
            if page_number <= 0 or page_number > len(writer.pages):
                continue
            if page_number > len(plumber_pdf.pages):
                continue

            page = plumber_pdf.pages[page_number - 1]
            used_rects = used_rects_by_page.setdefault(page_number, [])
            render_stats["line_items_targeted"] += 1
            rect = _find_match_rect(page, row.get("a_desc", ""), used_rects)
            if rect is None:
                comment = _comment_for_row(row)
                page_obj = writer.pages[page_number - 1]
                page_width = float(page_obj.mediabox.width)
                page_height = float(page_obj.mediabox.height)
                room_anchor_rect = _find_room_anchor_rect(page, str(row.get("room_a") or ""))
                if room_anchor_rect is not None:
                    note_rect = _note_rect_near_line(room_anchor_rect, page_width, page_height, note_size=14.0, gap=4.0)
                    render_stats["anchored_room_notes_added"] += 1
                else:
                    fallback_idx = fallback_note_count_by_page.get(page_number, 0)
                    note_rect = _fallback_note_rect(page_width, page_height, fallback_idx)
                    fallback_note_count_by_page[page_number] = fallback_idx + 1
                note = Text(
                    rect=note_rect,
                    text=comment,
                    open=False,
                    title_bar="Ciridae",
                )
                note[NameObject("/Subj")] = TextStringObject(f"{status.upper()}_UNLOCATED")
                note[NameObject("/C")] = _status_note_color_array(status)
                writer.add_annotation(page_number - 1, note)
                popup = Popup(
                    rect=(
                        max(8.0, note_rect[0] - 12.0),
                        max(8.0, note_rect[1] - 8.0),
                        max(220.0, min(page_width - 8.0, note_rect[0] + 300.0)),
                        max(150.0, min(page_height - 8.0, note_rect[1] + 170.0)),
                    ),
                    parent=note,
                    open=False,
                )
                writer.add_annotation(page_number - 1, popup)
                render_stats["unlocated_notes_added"] += 1
                continue

            used_rects.append(rect)

            comment = _comment_for_row(row)
            quad_points = _quad_points_for_rect(rect)
            highlight = Highlight(
                rect=rect,
                quad_points=quad_points,
                highlight_color=_status_highlight_color(status),
            )
            highlight[NameObject("/Contents")] = TextStringObject(comment)
            highlight[NameObject("/T")] = TextStringObject("Ciridae")
            highlight[NameObject("/Subj")] = TextStringObject(status.upper())
            writer.add_annotation(page_number - 1, highlight)
            render_stats["highlights_added"] += 1

            # Add a visible comment-note icon next to the matched line.
            page_obj = writer.pages[page_number - 1]
            page_width = float(page_obj.mediabox.width)
            page_height = float(page_obj.mediabox.height)
            note_rect = _note_rect_near_line(rect, page_width, page_height)
            note = Text(
                rect=note_rect,
                text=comment,
                open=False,
                title_bar="Ciridae",
            )
            note[NameObject("/Subj")] = TextStringObject(status.upper())
            note[NameObject("/C")] = _status_note_color_array(status)
            writer.add_annotation(page_number - 1, note)
            popup_rect = (
                max(8.0, min(note_rect[2] + 8.0, page_width - 280.0)),
                max(8.0, min(note_rect[1] - 8.0, page_height - 180.0)),
                max(220.0, min(note_rect[2] + 280.0, page_width - 8.0)),
                max(150.0, min(note_rect[1] + 150.0, page_height - 8.0)),
            )
            popup = Popup(
                rect=popup_rect,
                parent=note,
                open=False,
            )
            writer.add_annotation(page_number - 1, popup)
            render_stats["inline_notes_added"] += 1

        # Add nugget summary notes only on page 1 to avoid random notes scattered across many pages.
        nugget_rows: list[dict] = []
        seen_b_item_ids: set[int] = set()
        for row in rows_sorted:
            if (row.get("status") or "").lower() != "nugget":
                continue
            b_item_id = row.get("b_item_id")
            if isinstance(b_item_id, int):
                if b_item_id in seen_b_item_ids:
                    continue
                seen_b_item_ids.add(b_item_id)
            nugget_rows.append(row)

        if nugget_rows:
            nugget_rows = sorted(
                nugget_rows,
                key=lambda r: (
                    (r.get("room_b") or "").lower(),
                    (r.get("b_desc") or "").lower(),
                ),
            )
            page_number = 1
            page_obj = writer.pages[page_number - 1]
            page_width = float(page_obj.mediabox.width)
            page_height = float(page_obj.mediabox.height)

            chunk_size = 8
            for chunk_index, start in enumerate(range(0, len(nugget_rows), chunk_size)):
                chunk = nugget_rows[start : start + chunk_size]
                lines = [_nugget_line(r) for r in chunk]
                text = "Insurance-only nuggets:\n" + "\n".join(f"- {line}" for line in lines)

                rect_width = min(320.0, max(240.0, page_width * 0.48))
                line_count = len(lines) + 1
                rect_height = min(220.0, max(86.0, 22.0 + (line_count * 14.0)))
                x1 = page_width - 10.0
                x0 = max(10.0, x1 - rect_width)
                y1 = page_height - 18.0 - (chunk_index * (rect_height + 8.0))
                y0 = y1 - rect_height
                if y0 < 12.0:
                    break

                note = Text(
                    rect=(x0, y0, x1, y1),
                    text=text,
                    open=False,
                    title_bar="Ciridae",
                )
                note[NameObject("/Subj")] = TextStringObject("NUGGET")
                note[NameObject("/C")] = _status_note_color_array("nugget")
                writer.add_annotation(page_number - 1, note)
                render_stats["nugget_summary_notes"] += 1
                popup = Popup(
                    rect=(
                        max(8.0, x0 - 10.0),
                        max(8.0, y0 - 10.0),
                        min(page_width - 8.0, x0 + min(380.0, rect_width + 40.0)),
                        min(page_height - 8.0, y0 + min(260.0, rect_height + 40.0)),
                    ),
                    parent=note,
                    open=False,
                )
                writer.add_annotation(page_number - 1, popup)

        render_stats["summary_page_appended"] = False

    with out.open("wb") as f:
        writer.write(f)
    return str(out), render_stats


def _build_summary_page(rows: list[dict], summary: dict) -> bytes | None:
    """Build a reportlab PDF page summarising nuggets and critical blue items."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
    except ImportError:
        return None

    buf = io.BytesIO()
    doc_rl = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=4)
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=10)
    sub_style = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#555555"))

    content = []
    content.append(Paragraph("Ciridae — Comparison Summary", title_style))

    # Status breakdown
    green_n = int(summary.get("green", 0))
    orange_n = int(summary.get("orange", 0))
    blue_n = int(summary.get("blue", 0))
    nugget_n = int(summary.get("nugget", 0))
    total_a = summary.get("total_a")
    total_b = summary.get("total_b")

    stat_data = [
        ["Status", "Count", "Meaning"],
        ["Green", str(green_n), "Exact match (scope + metadata within \u00b12%)"],
        ["Orange", str(orange_n), "Same scope; metadata differs beyond \u00b12%"],
        ["Blue", str(blue_n), "JDR-only: not found in insurance estimate"],
        ["Nuggets", str(nugget_n), "Insurance-only: present in insurer\u2019s scope, not in JDR"],
    ]
    if total_a is not None and total_b is not None:
        stat_data.append(["Dollar gap", _money(abs(total_a - total_b)), f"JDR total {_money(total_a)} vs Insurance {_money(total_b)}"])

    stat_colors = {
        1: colors.HexColor("#bbf7d0"),
        2: colors.HexColor("#fed7aa"),
        3: colors.HexColor("#bfdbfe"),
        4: colors.HexColor("#fef08a"),
    }
    stat_table = Table(stat_data, colWidths=[1.0 * inch, 0.7 * inch, 5.0 * inch])
    stat_table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_idx, bg in stat_colors.items():
        stat_table_style.append(("BACKGROUND", (0, row_idx), (0, row_idx), bg))
    stat_table.setStyle(TableStyle(stat_table_style))
    content.append(stat_table)

    # Nuggets table
    nugget_rows = [r for r in rows if (r.get("status") or "").lower() == "nugget"]
    seen_nugget = set()
    unique_nuggets: list[dict] = []
    for r in nugget_rows:
        bid = r.get("b_item_id")
        key = bid if bid is not None else (r.get("room_b", ""), r.get("b_desc", ""))
        if key not in seen_nugget:
            seen_nugget.add(key)
            unique_nuggets.append(r)
    unique_nuggets.sort(key=lambda r: ((r.get("room_b") or "").lower(), (r.get("b_desc") or "").lower()))

    if unique_nuggets:
        content.append(Spacer(1, 8))
        content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
        content.append(Paragraph("Insurance-Only Items (Nuggets)", h2_style))
        content.append(Paragraph(
            "These items appear in the insurance estimate but are absent from the JDR. "
            "JDR can often add them directly since the insurer has already agreed to them.",
            sub_style,
        ))
        content.append(Spacer(1, 4))
        nug_data = [["Room (Insurance)", "Description", "Amount"]]
        for r in unique_nuggets:
            nug_data.append([
                Paragraph(_cut(r.get("room_b", ""), 30), cell_style),
                Paragraph(_cut(r.get("b_desc", ""), 90), cell_style),
                _money(r.get("b_amt")),
            ])
        nug_table = Table(nug_data, colWidths=[1.4 * inch, 4.3 * inch, 1.0 * inch])
        nug_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef9c3")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUND", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fefce8")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fde68a")),
            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        content.append(nug_table)

    try:
        doc_rl.build(content)
        return buf.getvalue()
    except Exception:
        return None
