"""Export the experiment note and selected measured plots as a standalone PDF."""

import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline(value):
    value = escape(value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    return re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    styles["BodyText"].spaceAfter = 7
    styles.add(ParagraphStyle("Cell", fontSize=6.5, leading=8))
    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
        title="Decentralized merge-aware LoRA serving: implementation and evaluation",
        author="Spindle experiment report",
    )
    lines = args.note.read_text().splitlines()
    flow = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            flow.append(
                Paragraph(
                    inline(line[level:].strip()), styles[f"Heading{min(level, 3)}"]
                )
            )
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                row = lines[index].strip("|").split("|")
                if not all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in row):
                    rows.append(
                        [
                            Paragraph(inline(cell.strip()), styles["Cell"])
                            for cell in row
                        ]
                    )
                index += 1
            table = Table(
                rows, colWidths=[doc.width / len(rows[0])] * len(rows[0]), repeatRows=1
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            flow.extend([table, Spacer(1, 10)])
        else:
            paragraph = []
            while (
                index < len(lines)
                and lines[index].strip()
                and not lines[index].startswith(("#", "|"))
            ):
                paragraph.append(lines[index].strip())
                index += 1
            flow.append(Paragraph(inline(" ".join(paragraph)), styles["BodyText"]))
    for path in args.plots:
        flow.append(PageBreak())
        flow.append(
            Paragraph(escape(path.parent.name + ": " + path.stem), styles["Heading2"])
        )
        figure = Image(str(path))
        ratio = min(
            doc.width / figure.imageWidth, (doc.height - 70) / figure.imageHeight
        )
        figure.drawWidth = figure.imageWidth * ratio
        figure.drawHeight = figure.imageHeight * ratio
        flow.append(figure)
        flow.append(Paragraph(escape(str(path)), styles["Cell"]))

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            doc.leftMargin,
            0.3 * inch,
            "Spindle experimental serving evaluation | 15 October 2026",
        )
        canvas.drawRightString(A4[0] - doc.rightMargin, 0.3 * inch, str(document.page))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
