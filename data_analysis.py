import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random


#extarct tables from pdf and csv

def extract_from_tables_pdf(pdf_docs):
    all_tables = []
    for pdf_file in pdf_docs:
        if pdf_file.name.endswith(".pdf"):
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                all_tables.append(df)

        elif pdf_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(pdf_file)
                all_tables.append(df)
            except Exception as e:
                st.warning(f"Couldn't read {pdf_file.name} as CSV. Error: {e}")
    return all_tables



#anayze and visualize the extracted tables
def analyze(tables):
    st.subheader("Data Analysis & Visualizations")

    if not tables:
        st.warning("No tables found in the uploaded PDF files.")
        return

    for i, df in enumerate(tables):
        st.markdown(f"### Table {i+1}")
        st.dataframe(df)

        for col in df.columns:      #tables to numeric conversion
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                continue

        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

        if len(numeric_cols) >= 2:
            selectable_cols = list(numeric_cols[1:])    #ensures columns is a list
            random_cols=random.sample(selectable_cols, 2)

            st.write(f"### Scatter Plot between `{random_cols[0]}` and `{random_cols[1]}`")
            fig = px.scatter(df, x=random_cols[0], y=random_cols[1])

            st.plotly_chart(fig)

        elif len(numeric_cols) == 1:
            # Use column with highest standard deviation
            col_to_plot = df[numeric_cols].std().idxmax()

            st.write(f"Bar Plot: `{col_to_plot}` (most variable column)")
            fig = px.bar(df, x=df.index, y=col_to_plot, title=f"Bar Chart of {col_to_plot}")
            st.plotly_chart(fig)

        # Scatter Matrix (Pairwise Plots)
        if len(numeric_cols) >= 2:
            st.markdown("#### 🔁 Scatter Matrix (Pairwise Numeric Columns)")
            fig = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix")
            st.plotly_chart(fig)

        # Correlation Heatmap
        if len(numeric_cols) >= 2:
            st.markdown("#### 🔥 Correlation Heatmap")

            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Multi-Line Plot (Assumes index or column represents time/order)
        if len(numeric_cols) >= 2:
            st.markdown("#### 📈 Multi-Line Trend Plot")
            # Use index as X if it looks like a sequence (or add time col detection later)
            fig = px.line(df[numeric_cols], title="Multi-Line Plot of Numeric Columns")
            st.plotly_chart(fig)

        else:
            st.info("No numeric data available for plotting in this table.")