from io import BytesIO
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape


PURPLE = colors.HexColor("#660066")


def _parse_dt(raw: Any):
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone()
    except Exception:
        return None


def _fmt_dt(raw: Any) -> str:
    dt = _parse_dt(raw)
    if dt is None:
        return "—"
    return dt.strftime("%d/%m/%Y %H:%M")


def _fmt_num(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        return f"{v:.1f}{suffix}"
    except Exception:
        return "—"


def _safe_text(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    return escape(text) if text else "—"


def _latest_dt(entries: list[dict]) -> datetime | None:
    dates = []
    for e in entries:
        d = _parse_dt(e.get("created_at"))
        if d:
            dates.append(d)
    return max(dates) if dates else None


def _build_rom_rows(rom_history: list[dict]) -> list[list[str]]:
    rows = []
    for row in rom_history:
        rows.append([
            _safe_text(row.get("movement")),
            _safe_text(row.get("side")),
            _fmt_num(row.get("max_angle"), "°"),
            _fmt_num(row.get("rom"), "°"),
            _fmt_dt(row.get("created_at")),
        ])
    return rows


def _build_gait_rows(gait_history: list[dict]) -> list[list[str]]:
    rows = []
    for entry in gait_history:
        results = entry.get("results") or {}
        if not isinstance(results, dict):
            results = {}

        rows.append([
            _fmt_dt(entry.get("created_at")),
            _safe_text(entry.get("view")).title(),
            _fmt_num(results.get("cadence"), " spm"),
            _fmt_num(results.get("speed"), " m/s"),
            _fmt_num(results.get("symmetry"), "%"),
            _fmt_num(results.get("confidence"), "%"),
        ])
    return rows


def _build_note_rows(notes: list[dict]) -> list[list[str]]:
    rows = []
    for n in notes:
        note_text = _safe_text(n.get("note"))
        if len(note_text) > 180:
            note_text = note_text[:180] + "..."
        rows.append([
            _fmt_dt(n.get("created_at")),
            _safe_text(n.get("title")),
            note_text,
        ])
    return rows


def _make_table(data: list[list[str]], col_widths: list[int], header_bg=PURPLE):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEADING", (0, 0), (-1, -1), 11),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7D7D7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8FB")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    return table


def build_progress_report_pdf(
    client_id: str,
    rom_history: list[dict],
    gait_history: list[dict],
    notes: list[dict],
    generated_at: datetime | None = None,
) -> bytes:
    generated_at = generated_at or datetime.now().astimezone()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallGrey",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#555555"),
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=15,
            textColor=PURPLE,
            spaceAfter=6,
        )
    )

    story = []

    story.append(Paragraph("StretchMasters Progress Report", styles["Title"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"Patient: {_safe_text(client_id)}", styles["SmallGrey"]))
    story.append(Paragraph(f"Generated: {_fmt_dt(generated_at.isoformat())}", styles["SmallGrey"]))
    story.append(Spacer(1, 8))

    summary_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["ROM sessions", str(len(rom_history)), "Gait sessions", str(len(gait_history))],
        ["Notes", str(len(notes)), "Latest ROM", _fmt_dt(_latest_dt(rom_history))],
        ["Latest gait", _fmt_dt(_latest_dt(gait_history)), "Latest note", _fmt_dt(_latest_dt(notes))],
    ]
    story.append(_make_table(summary_data, [38 * mm, 52 * mm, 38 * mm, 52 * mm]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("ROM history", styles["SectionTitle"]))
    if rom_history:
        rom_data = [["Movement", "Side", "Max Angle", "ROM", "Date"]] + _build_rom_rows(rom_history)
        story.append(_make_table(rom_data, [42 * mm, 22 * mm, 28 * mm, 22 * mm, 38 * mm]))
    else:
        story.append(Paragraph("No ROM history available for this patient.", styles["SmallGrey"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Gait history", styles["SectionTitle"]))
    if gait_history:
        gait_data = [["Date", "View", "Cadence", "Speed", "Symmetry", "Confidence"]] + _build_gait_rows(gait_history)
        story.append(_make_table(gait_data, [34 * mm, 18 * mm, 24 * mm, 22 * mm, 24 * mm, 24 * mm]))
        latest_gait = gait_history[0] if gait_history else None
        if latest_gait:
            results = latest_gait.get("results") or {}
            if isinstance(results, dict):
                summary = (results.get("summary") or results.get("message") or "").strip()
                if summary:
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(f"<b>Latest gait summary:</b> {_safe_text(summary)}", styles["SmallGrey"]))
    else:
        story.append(Paragraph("No gait history available for this patient.", styles["SmallGrey"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Clinical notes", styles["SectionTitle"]))
    if notes:
        note_data = [["Date", "Title", "Note excerpt"]] + _build_note_rows(notes)
        story.append(_make_table(note_data, [30 * mm, 40 * mm, 90 * mm]))
    else:
        story.append(Paragraph("No notes available for this patient.", styles["SmallGrey"]))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "This report is generated from live backend history and clinical notes.",
            styles["SmallGrey"],
        )
    )

    def _footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(doc_obj.leftMargin, 10 * mm, f"StretchMasters - {_safe_text(client_id)}")
        canvas.drawRightString(
            A4[0] - doc_obj.rightMargin,
            10 * mm,
            f"Page {canvas.getPageNumber()}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
