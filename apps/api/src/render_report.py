from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

def _money(x):
    if x is None:
        return "-"
    return f"${x:,.2f}"

def render_report_pdf(
    out_path: str,
    run_id: str,
    rows: list[dict],
    summary: dict,
) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out), pagesize=letter)
    width, height = letter

    def header(title: str):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.75 * inch, height - 0.75 * inch, title)
        c.setFont("Helvetica", 10)
        c.drawString(0.75 * inch, height - 0.95 * inch, f"run_id: {run_id}")
        c.drawString(0.75 * inch, height - 1.10 * inch, f"green: {summary['green']}  orange: {summary['orange']}  blue: {summary['blue']}")
        c.drawString(0.75 * inch, height - 1.25 * inch, f"A total (matched): {_money(summary.get('total_a'))}   B total (matched): {_money(summary.get('total_b'))}")
        c.line(0.75 * inch, height - 1.35 * inch, width - 0.75 * inch, height - 1.35 * inch)

    def new_page(title: str):
        c.showPage()
        header(title)

    # Page 1 summary
    header("Estimate Comparison Report")
    y = height - 1.7 * inch
    c.setFont("Helvetica", 10)
    c.drawString(0.75 * inch, y, "This report groups items by room mapping and labels:")
    y -= 14
    c.drawString(0.95 * inch, y, "GREEN  = matched + within ±2%")
    y -= 14
    c.drawString(0.95 * inch, y, "ORANGE = matched but cost differs >2% or scope differs")
    y -= 14
    c.drawString(0.95 * inch, y, "BLUE   = contractor-only (nugget)")

    # Group rows by room pair
    def room_key(r): 
        return (r["room_a"], r["room_b"])

    rows_sorted = sorted(rows, key=lambda r: (r["room_a"], r["room_b"], r["status"]))
    current_room = None

    # Start detail pages
    new_page("Room-by-room detail")
    y = height - 1.7 * inch

    c.setFont("Helvetica-Bold", 11)

    for r in rows_sorted:
        rk = room_key(r)
        if rk != current_room:
            current_room = rk
            if y < 1.5 * inch:
                new_page("Room-by-room detail")
                y = height - 1.7 * inch
            c.setFont("Helvetica-Bold", 12)
            c.drawString(0.75 * inch, y, f"{rk[0]}  ↔  {rk[1]}")
            y -= 16
            c.setFont("Helvetica", 9)
            c.drawString(0.75 * inch, y, "status | A description (A$)  ->  B description (B$)   [why]")
            y -= 12

        if y < 1.2 * inch:
            new_page("Room-by-room detail")
            y = height - 1.7 * inch

        status = r["status"].upper()
        a_desc = (r.get("a_desc") or "").strip()
        b_desc = (r.get("b_desc") or "").strip()
        why = (r.get("rationale") or "").strip()

        # tighten long text
        def cut(s, n):
            return s if len(s) <= n else s[: n - 3] + "..."

        line = f"{status:6} | {cut(a_desc, 56)} ({_money(r.get('a_amt'))}) -> {cut(b_desc, 48)} ({_money(r.get('b_amt'))})"
        c.setFont("Helvetica", 9)
        c.drawString(0.75 * inch, y, line)
        y -= 11
        if why:
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(0.95 * inch, y, cut(why, 115))
            y -= 11

    c.save()
    return str(out)
