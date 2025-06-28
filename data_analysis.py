import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import tempfile
import kaleido
from export import export_file


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
    st.subheader("📊 Data Analysis & Enhanced Visualizations")
    if not tables:
        st.warning("No tables found in the uploaded files.")
        return

    #paths for storing plots
    if "plot_paths" not in st.session_state:
        st.session_state.plot_paths=[]

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir=tempfile.mkdtemp()


    for i, df in enumerate(tables):
        st.markdown(f"### 📄 Table {i+1}")
        st.dataframe(df)

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                continue

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) >= 2:
            selectable_cols = list(numeric_cols[1:])
            if len(selectable_cols) >= 2:
                random_cols = random.sample(selectable_cols, 2)

                st.write(f"### 🎯 Scatter Plot: `{random_cols[0]}` vs `{random_cols[1]}`")
                fig = px.scatter(
                    df, x=random_cols[0], y=random_cols[1],
                    color=random_cols[0],
                    symbol=random_cols[1],
                    size=random_cols[1],
                    hover_data=df.columns,
                    title=f"Scatter Plot between {random_cols[0]} and {random_cols[1]}",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                path = os.path.join(st.session_state.temp_dir, f"table_{i}_scatter.png")
                fig.write_image(path)
                st.session_state.plot_paths.append(path)

        elif len(numeric_cols) == 1:
            col_to_plot = df[numeric_cols].std().idxmax()

            st.write(f"### 📊 Bar Plot: `{col_to_plot}`")
            fig = px.bar(
                df.sort_values(by=col_to_plot, ascending=False),
                x=df.index,
                y=col_to_plot,
                color=col_to_plot,
                title=f"Bar Chart of {col_to_plot}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_bar.png")
            fig.write_image(path)
            st.session_state.plot_paths.append(path)

        # Scatter Matrix
        if len(numeric_cols) >= 3:
            st.markdown("#### 🔁 Scatter Matrix (Pairwise)")
            fig = px.scatter_matrix(
                df,
                dimensions=numeric_cols,
                color=numeric_cols[0],
                title="Scatter Matrix of Numeric Columns",
                template="simple_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_matrix.png")
            fig.write_image(path)
            st.session_state.plot_paths.append(path)

        # Correlation Heatmap
        if len(numeric_cols) >= 2:
            st.markdown("#### 🌡️ Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr,
                annot=True,
                cmap="RdYlGn",
                fmt=".2f",
                linewidths=0.5,
                linecolor='gray',
                cbar_kws={"shrink": .75}
            )
            ax.set_title("Correlation Matrix", fontsize=14)
            st.pyplot(fig)

            path = os.path.join(st.session_state.temp_dir, f"table_{i}_heatmap.png")
            fig.savefig(path)
            plt.close(fig)
            st.session_state.plot_paths.append(path)

        # Multi-Line Trend Plot
        if len(numeric_cols) >= 2:
            st.markdown("#### 📈 Multi-Line Trend Plot")
            fig = px.line(
                df[numeric_cols],
                title="Multi-Line Trend of Numeric Columns",
                markers=True,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_line.png")
            fig.write_image(path)
            st.session_state.plot_paths.append(path)

        else:
            st.info("No numeric data available for plotting in this table.")
