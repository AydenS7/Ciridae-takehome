from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def _money(x):
    if x is None:
        return "-"
    return f"${x:,.2f}"


def _cut(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _status_color(status_raw: str):
    status_raw = (status_raw or "").lower()
    if status_raw == "green":
        return colors.green
    if status_raw == "orange":
        return colors.orange
    if status_raw == "blue":
        return colors.blue
    return colors.black


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
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.75 * inch, height - 0.75 * inch, title)
        c.setFont("Helvetica", 10)
        c.drawString(0.75 * inch, height - 0.95 * inch, f"run_id: {run_id}")
        c.drawString(
            0.75 * inch,
            height - 1.10 * inch,
            f"green: {summary['green']}  orange: {summary['orange']}  blue: {summary['blue']}",
        )
        c.drawString(
            0.75 * inch,
            height - 1.25 * inch,
            f"A total (matched): {_money(summary.get('total_a'))}   B total (matched): {_money(summary.get('total_b'))}",
        )
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
    c.setFillColor(colors.green)
    c.drawString(0.95 * inch, y, "GREEN  = matched + within ±2%")
    y -= 14
    c.setFillColor(colors.orange)
    c.drawString(0.95 * inch, y, "ORANGE = matched but cost differs >2% or scope differs")
    y -= 14
    c.setFillColor(colors.blue)
    c.drawString(0.95 * inch, y, "BLUE   = contractor-only (nugget)")
    c.setFillColor(colors.black)

    # Group rows by room pair
    rows_sorted = sorted(rows, key=lambda r: (r.get("room_a", ""), r.get("room_b", ""), r.get("status", "")))
    current_room = None

    # Start detail pages
    new_page("Room-by-room detail")
    y = height - 1.7 * inch

    for r in rows_sorted:
        rk = (r.get("room_a", ""), r.get("room_b", ""))
        if rk != current_room:
            current_room = rk
            if y < 1.5 * inch:
                new_page("Room-by-room detail")
                y = height - 1.7 * inch
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(0.75 * inch, y, f"{rk[0]}  ↔  {rk[1]}")
            y -= 16
            c.setFont("Helvetica", 9)
            c.drawString(0.75 * inch, y, "status | A description (A$)  ->  B description (B$)   [why]")
            y -= 12

        if y < 1.2 * inch:
            new_page("Room-by-room detail")
            y = height - 1.7 * inch

        status_raw = (r.get("status") or "").lower()
        status = status_raw.upper()

        a_desc = _cut(r.get("a_desc", ""), 56)
        b_desc = _cut(r.get("b_desc", ""), 48)
        why = (r.get("rationale") or "").strip()

        line = f"{status:6} | {a_desc} ({_money(r.get('a_amt'))}) -> {b_desc} ({_money(r.get('b_amt'))})"

        # Colored square marker (very visible)
        col = _status_color(status_raw)
        c.setFillColor(col)
        c.rect(0.75 * inch, y - 2, 6, 6, fill=1, stroke=0)

        # Colored status text, rest black for readability
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(col)
        c.drawString(0.75 * inch + 10, y, f"{status:6}")

        c.setFont("Helvetica", 9)
        c.setFillColor(colors.black)
        c.drawString(0.75 * inch + 55, y, line.split("|", 1)[1].strip())

        y -= 11
        if why:
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(0.95 * inch, y, _cut(why, 115))
            y -= 11

    c.save()
    return str(out)
