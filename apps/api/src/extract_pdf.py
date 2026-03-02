from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pdfplumber

ROOM_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &/\-]{2,}$")  # conservative: ALL CAPS headers
MONEY_RE = re.compile(r"(?<!\w)\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)(?!\w)")

def _parse_amount(text: str) -> Optional[float]:
    """
    Heuristic: pick the last money-like token on the line.
    Works well for 'desc .... 1,234.56' style rows.
    """
    matches = list(MONEY_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None

def _clean_line(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass
class ExtractedLine:
    page: int
    room: str
    description: str
    amount: Optional[float]

def extract_lines(pdf_path: str) -> list[ExtractedLine]:
    """
    Extract lines page-by-page using pdfplumber.
    Tracks a 'current room' based on uppercase section headers.
    """
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
                if not line:
                    continue

                # Update current room if this looks like a section header
                if ROOM_HEADER_RE.match(line) and len(line) <= 60:
                    current_room = line
                    continue

                # Skip obvious noise
                if line.lower() in {"page", "total"}:
                    continue

                amt = _parse_amount(line)
                results.append(
                    ExtractedLine(
                        page=page_index,
                        room=current_room,
                        description=line,
                        amount=amt,
                    )
                )

    return results
