from __future__ import annotations

from typing import Literal
import pdfplumber

from .llm_client import get_client
from .llm_schemas import ExtractPageResult

Doc = Literal["A", "B"]

SYSTEM_PROMPT = """You extract ONLY estimate line items from insurance/contractor repair estimates.

Rules:
- Ignore cover pages, contact info, disclaimers, claim numbers, phone numbers, dates, policy text, and summary guides.
- Output ONLY real estimate rows (things that would be billed), not section/table headers.
- Assign each item to a room/area/section based on the nearest heading on the page.
- If an item has no clear total cost, set total=null (still include item if it's clearly an estimate row).
- confidence should reflect how sure you are it is a real line item row.
"""

def _chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    # simple, safe chunker: split on lines
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

def extract_pdf_via_llm(pdf_path: str, doc: Doc) -> list[ExtractPageResult]:
    client = get_client()
    results: list[ExtractPageResult] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text = text.strip()
            if not text:
                continue

            # Chunk very large pages to avoid overloading context
            chunks = _chunk_text(text)

            page_items = []
            for chunk in chunks:
                user = f"""Document: {doc}
Page: {page_index}

Extract line items from this page text:

---BEGIN PAGE TEXT---
{chunk}
---END PAGE TEXT---
"""
                # Structured outputs via Pydantic parsing
                parsed = client.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    response_format=ExtractPageResult,
                ).choices[0].message.parsed

                # parsed.items may include low confidence junk; filter lightly here
                page_items.extend([it for it in parsed.items if it.confidence >= 0.5])

            results.append(ExtractPageResult(doc=doc, page=page_index, items=page_items))

    return results
