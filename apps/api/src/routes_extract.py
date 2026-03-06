"""Endpoints for extracting line items from uploaded proposals and listing results."""

from concurrent.futures import ThreadPoolExecutor
import re

from fastapi import APIRouter, HTTPException
import pdfplumber
from sqlalchemy import delete
from time import perf_counter

from .db import SessionLocal
from .models import Run
from .models_items import LineItem
from .extract_pdf_llm import extract_pdf_via_llm

router = APIRouter(prefix="/runs", tags=["runs"])

_PAGE_INDEX_RE = re.compile(r"(?m)^\s*(\d{1,4})\s*[\.\)]\s+")
_DESC_INDEX_RE = re.compile(r"^\s*(\d{1,4})\s*[\.\)]\s+")


def _unique_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _source_indices_by_page(pdf_path: str) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text(x_tolerance=2, y_tolerance=2) or "").strip()
            if not text:
                continue
            nums = [int(match.group(1)) for match in _PAGE_INDEX_RE.finditer(text)]
            nums = _unique_preserve_order(nums)
            if nums:
                indices[page_number] = nums
    return indices


def _extracted_indices_by_page(doc_pages: list) -> dict[int, list[int]]:
    indices: dict[int, list[int]] = {}
    for page_result in doc_pages:
        nums: list[int] = []
        for item in page_result.items:
            match = _DESC_INDEX_RE.match((item.description or "").strip())
            if match:
                nums.append(int(match.group(1)))
        nums = _unique_preserve_order(nums)
        if nums:
            indices[int(page_result.page)] = nums
    return indices


def _index_coverage_report(pdf_path: str, doc_pages: list) -> dict:
    source = _source_indices_by_page(pdf_path)
    extracted = _extracted_indices_by_page(doc_pages)

    pages: list[dict] = []
    expected_total = 0
    covered_total = 0
    pages_full = 0

    for page_number in sorted(source.keys()):
        expected = source.get(page_number, [])
        got = extracted.get(page_number, [])
        expected_set = set(expected)
        got_set = set(got)
        missing = [n for n in expected if n not in got_set]
        extras = [n for n in got if n not in expected_set]
        covered = len(expected) - len(missing)
        expected_total += len(expected)
        covered_total += covered
        is_full = len(missing) == 0
        if is_full:
            pages_full += 1

        pages.append(
            {
                "page": page_number,
                "first_index": expected[0] if expected else None,
                "last_index": expected[-1] if expected else None,
                "expected_count": len(expected),
                "extracted_count": len(got),
                "covered_count": covered,
                "missing_indices": missing[:40],
                "extra_indices": extras[:20],
                "is_full": is_full,
            }
        )

    coverage_pct = round((covered_total / expected_total) * 100.0, 1) if expected_total > 0 else None
    problem_pages = [p for p in pages if not p["is_full"]]
    return {
        "pages_with_indices": len(source),
        "pages_full": pages_full,
        "pages_incomplete": max(0, len(source) - pages_full),
        "expected_indices_total": expected_total,
        "covered_indices_total": covered_total,
        "missing_indices_total": max(0, expected_total - covered_total),
        "coverage_pct": coverage_pct,
        "problem_pages": problem_pages[:25],
        "pages": pages,
    }

@router.post("/{run_id}/extract")
def extract_run(run_id: str):
    t0 = perf_counter()
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")

        db.execute(delete(LineItem).where(LineItem.run_id == run_id))
        db.commit()

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(extract_pdf_via_llm, run.proposal_a_path, "A")
            fut_b = ex.submit(extract_pdf_via_llm, run.proposal_b_path, "B")
            a_pages, a_stats = fut_a.result()
            b_pages, b_stats = fut_b.result()

        def persist(doc_pages):
            n = 0
            seen = set()
            for page_result in doc_pages:
                for it in page_result.items:
                    key = (
                        page_result.doc,
                        page_result.page,
                        it.room.strip().lower(),
                        it.description.strip().lower(),
                        round(it.total or 0.0, 2) if it.total is not None else None,
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    db.add(LineItem(
                        run_id=run_id,
                        doc=page_result.doc,
                        page=page_result.page,
                        room=it.room,
                        description=it.description,
                        quantity=it.quantity,
                        unit=it.unit,
                        unit_price=it.unit_price,
                        amount=it.total,
                    ))
                    n += 1
            return n

        n_a = persist(a_pages)
        n_b = persist(b_pages)
        db.commit()
        coverage_a = _index_coverage_report(run.proposal_a_path, a_pages)
        coverage_b = _index_coverage_report(run.proposal_b_path, b_pages)

        elapsed_ms = int((perf_counter() - t0) * 1000)
        return {
            "run_id": run_id,
            "extracted": {"A": n_a, "B": n_b},
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "docs": {
                    "A": {**a_stats, "index_coverage": coverage_a},
                    "B": {**b_stats, "index_coverage": coverage_b},
                },
            },
        }

@router.get("/{run_id}/items")
def list_items(run_id: str, doc: str | None = None, limit: int = 200):
    with SessionLocal() as db:
        q = db.query(LineItem).filter(LineItem.run_id == run_id)
        if doc:
            q = q.filter(LineItem.doc == doc)
        items = q.order_by(LineItem.doc, LineItem.page, LineItem.id).limit(limit).all()

        return [
            {
                "id": it.id,
                "doc": it.doc,
                "page": it.page,
                "room": it.room,
                "description": it.description,
                "amount": it.amount,
            }
            for it in items
        ]
