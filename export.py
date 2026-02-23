from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch
import tempfile
import streamlit as st
import os


def export_file(tables):
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmpfile.name, pagesize=landscape(A4))
    elements = []

    styles = getSampleStyleSheet()

    for i, df in enumerate(tables):
        elements.append(Paragraph(f"<b>Table {i+1}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

        df = df.head(20).fillna("").astype(str)

        data = [df.columns.tolist()] + df.values.tolist()

        table = Table(data, repeatRows=1)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))

        # Add plots
        for path in st.session_state.get("plot_paths", []):
            if path.endswith(".png") and os.path.exists(path):
                elements.append(Image(path, width=8*inch, height=4*inch))
                elements.append(Spacer(1, 0.5 * inch))

    doc.build(elements)
    return tmpfile.name