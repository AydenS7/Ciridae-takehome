from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pdfplumber

# Lines that are almost certainly table headers / noise, not rooms
ROOM_BANNED_EXACT = {
    "DESCRIPTION QTY RESET REMOVE REPLACE TAX O&P TOTAL",
    "DESCRIPTION QTY REMOVE REPLACE TAX O&P TOTAL",
    "DESCRIPTION QTY TAX O&P TOTAL",
}

# Room headers are usually short, often Title Case ("Main Level", "Kitchen") or ALL CAPS.
TITLE_CASE_ROOM_RE = re.compile(r"^[A-Z][a-z]+(?:[ /&\-][A-Z][a-z]+){0,6}$")
ALL_CAPS_ROOM_RE = re.compile(r"^[A-Z][A-Z0-9 /&\-]{2,}$")

# Money: prefer $ or comma/decimal formatting to avoid phone numbers / IDs / dates
MONEY_STRICT_RE = re.compile(r"(\$[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)")
MONEY_LOOSE_RE = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})+\.[0-9]{2})")  # must have comma AND cents
MONEY_CENTS_RE = re.compile(r"([0-9]+\.[0-9]{2})")  # fallback if row-like

# Skip lines that are clearly boilerplate
SKIP_SUBSTRINGS = [
    "state farm",
    "statefarmfireclaims",
    "this estimate is priced",
    "terms, conditions and limits",
    "please contact your claim",
    "provided for reference only",
    "cannot authorize any contractor",
]

def _clean_line(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _looks_like_room(line: str) -> bool:
    if not line:
        return False
    if line in ROOM_BANNED_EXACT:
        return False
    # too long to be a room
    if len(line) > 45:
        return False
    # avoid pure numbers
    if line.isdigit():
        return False
    return bool(TITLE_CASE_ROOM_RE.match(line) or ALL_CAPS_ROOM_RE.match(line))

def _is_noise(line: str) -> bool:
    low = line.lower()
    if any(s in low for s in SKIP_SUBSTRINGS):
        return True
    # page footer patterns
    if low.startswith("date:") and "page:" in low:
        return True
    # single token that is a number (page number)
    if re.fullmatch(r"\d{1,3}", line):
        return True
    return False

def _parse_amount_strict(line: str) -> Optional[float]:
    """
    Prefer a strict money token; otherwise allow cents-only if the line looks like a row
    (has multiple spaces / columns / quantity markers).
    """
    m = MONEY_STRICT_RE.search(line)
    if m:
        return float(m.group(1).replace("$", "").replace(",", ""))

    m = MONEY_LOOSE_RE.search(line)
    if m:
        return float(m.group(1).replace(",", ""))

    # last resort: if the line looks like a line-item row, allow X.XX at end
    row_like = ("  " in line) or bool(re.search(r"\b(EA|HR|LF|SF|SY|SQ|DA)\b", line))
    if row_like:
        cents = list(MONEY_CENTS_RE.finditer(line))
        if cents:
            raw = cents[-1].group(1)
            try:
                return float(raw)
            except ValueError:
                return None
    return None

@dataclass
class ExtractedLine:
    page: int
    room: str
    description: str
    amount: Optional[float]

def extract_lines(pdf_path: str) -> list[ExtractedLine]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)

    results: list[ExtractedLine] = []
    current_room = "(unknown)"

    with pdfplumber.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            raw_lines = [ln for ln in text.splitlines() if ln.strip()]

            for raw in raw_lines:
                line = _clean_line(raw)
                if not line or _is_noise(line):
                    continue

                # Update room
                if _looks_like_room(line):
                    current_room = line
                    continue

                # Parse money carefully
                amt = _parse_amount_strict(line)

                # Only keep "line items" that have an amount OR look like a scoped item row ("1. ...")
                looks_like_item = bool(re.match(r"^\d+\.\s", line))
                if amt is None and not looks_like_item:
                    continue

                results.append(
                    ExtractedLine(
                        page=page_index,
                        room=current_room,
                        description=line,
                        amount=amt,
                    )
                )

    return results
