import streamlit as st
import pandas as pd
import pdfplumber
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import tempfile
from PIL import Image
import io
import re
from export import export_file
import plotly.io as pio

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



def safe_filename(name, max_length=50):     #for avioding filename and runtime error
    """Sanitize a string to be used as a safe filename."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)[:max_length]


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
        ignore_cols = [col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()]    # Remove unwanted columns like 'id', 'unnamed'
        df = df.drop(columns=ignore_cols, errors='ignore')

        st.markdown(f"### 📄 Table {i+1}")
        st.dataframe(df)

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                continue


        numeric_cols = [     # Only select truly numeric columns with enough unique values
            col for col in df.select_dtypes(include=["int64", "float64"]).columns
            if df[col].nunique() > 5  # Exclude low-cardinality numerics (e.g., flags or IDs)
        ]
        
        def save_file_as_image(fig, path):
            try:
                pio.write_html(fig, file=path.replace(".png", ".html"), auto_open=False)
                st.session_state.plot_paths.append(path.replace(".png", ".html"))
            except Exception as e:
                st.warning(f"Couldn't export plot to image: {e}")

    #PLOTTINGS
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
                x_col=safe_filename(random_cols[0])
                y_col=safe_filename(random_cols[1])
                path = os.path.join(st.session_state.temp_dir, f"table_{i}_scatter_{x_col}_{y_col}.png")
                save_file_as_image(fig,path)

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
            col_bar=safe_filename(col_to_plot)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_bar_{col_bar}.png")
            save_file_as_image(fig , path)

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
            first_col = safe_filename(numeric_cols[0]) if numeric_cols else "matrix"
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_matrix_{first_col}.png")
            save_file_as_image(fig,path)

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
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_line_trend.png")
            save_file_as_image(fig,path)            

        # 📊 Categorical Columns (object or low unique count)
        cat_cols = [
            col for col in df.columns
            if df[col].dtype == 'object' or df[col].nunique() <= 5
        ]

        if cat_cols:
            st.markdown("#### 🧩 Categorical Columns Overview")
            for col in cat_cols:
                try:
                    fig = px.histogram(
                        df,
                        x=col,
                        color_discrete_sequence=["#FF6F61"],
                        title=f"Count Plot for '{col}'"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    safe_col=re.sub(r'[^a-zA-Z0-9_]','_',col)[:50]  #sanitize and truncate 
                    path = os.path.join(st.session_state.temp_dir, f"table_{i}_cat_{safe_col}.png")
                    save_file_as_image(fig,path)
                except Exception as e:
                    st.warning(f"⚠️ Skipped column '{col}': {e}")


        else:
            st.info("No numeric data available for plotting in this table.")
