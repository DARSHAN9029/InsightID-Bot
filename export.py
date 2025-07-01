from fpdf import FPDF
import os
import tempfile
import streamlit as st
from PIL import Image


def export_file(tables):
    pdf=FPDF()
    pdf.set_auto_page_break(auto=True , margin=15)

    for i, df in enumerate(tables):
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Table {i + 1}", ln=True)

        pdf.set_font("Arial", size=10)
        data = df.head(20).fillna("").astype(str).values.tolist()  # limit to 20 rows
        col_width = pdf.w / (len(df.columns) + 1)

        # Column headers
        for col_name in df.columns:
            pdf.cell(col_width, 10, col_name[:15], border=1)
        pdf.ln()

        # Rows
        for row in data:
            for cell in row:
                pdf.cell(col_width, 10, cell[:15], border=1)
            pdf.ln()

        for path in st.session_state.get("plot_paths",[]):
            if path.endswith(".png") and os.path.exists(path):
                pdf.add_page()
                pdf.image(path, w=pdf.w - 20)

    with tempfile.NamedTemporaryFile(delete=False , suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        return tmpfile.name
