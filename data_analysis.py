import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import tempfile
import re
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


def safe_filename(name, max_length=50):     #for avioding filename and runtime error
    """Sanitize a string to be used as a safe filename."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)[:max_length]


#anayze and visualize the extracted tables
def analyze(tables):
    st.subheader("📊 Data Analysis & Enhanced Visualizations")
    if not tables:
        st.warning("No tables found in the uploaded files.")
        return

    if "plot_paths" not in st.session_state:
        st.session_state.plot_paths = []

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

    for i, df in enumerate(tables):
        ignore_cols = [col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()]
        df = df.drop(columns=ignore_cols, errors='ignore')

        st.markdown(f"### 📄 Table {i+1}")
        st.dataframe(df)

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                continue

        numeric_cols = [
            col for col in df.select_dtypes(include=["int64", "float64"]).columns
            if df[col].nunique() > 5
        ]

        def save_plt(path):
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            st.session_state.plot_paths.append(path)

        if "Scatter Plot" in st.session_state.get("selected_plots",[]) and len(numeric_cols) >= 2:
            selectable_cols = list(numeric_cols[1:])
            if len(selectable_cols) >= 2:
                random_cols = random.sample(selectable_cols, 2)

                st.write(f"### 🎯 Scatter Plot: `{random_cols[0]}` vs `{random_cols[1]}`")
                plt.figure(figsize=(8, 6))
                plt.scatter(df[random_cols[0]], df[random_cols[1]], alpha=0.7, c='blue')
                plt.xlabel(random_cols[0])
                plt.ylabel(random_cols[1])
                plt.title(f"Scatter Plot between {random_cols[0]} and {random_cols[1]}")
                st.pyplot(plt)
                path = os.path.join(st.session_state.temp_dir, f"table_{i}_scatter_{safe_filename(random_cols[0])}_{safe_filename(random_cols[1])}.png")
                save_plt(path)

        elif "Bar Plot" in st.session_state.get("selected_plots",[]) and len(numeric_cols) == 1:
            col_to_plot = df[numeric_cols].std().idxmax()
            st.write(f"### 📊 Bar Plot: `{col_to_plot}`")
            plt.figure(figsize=(10, 6))
            df[col_to_plot].head(30).plot(kind='bar', color='orange')
            plt.title(f"Bar Chart of {col_to_plot}")
            plt.xlabel("Index")
            plt.ylabel(col_to_plot)
            st.pyplot(plt)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_bar_{safe_filename(col_to_plot)}.png")
            save_plt(path)

        if "Scatter Matrix" in st.session_state.get("selected_plots",[]) and len(numeric_cols) >= 3:
            st.markdown("#### 🔁 Scatter Matrix (Pairwise)")
            sns_plot = sns.pairplot(df[numeric_cols].dropna())
            st.pyplot(sns_plot)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_matrix_{safe_filename(numeric_cols[0])}.png")
            sns_plot.savefig(path)
            plt.close()
            st.session_state.plot_paths.append(path)

        if "Co-relation Heatmap" in st.session_state.get("selected_plots" , []) and len(numeric_cols) >= 2:
            st.markdown("#### 🌡️ Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5, linecolor='gray', cbar_kws={"shrink": .75})
            ax.set_title("Correlation Matrix", fontsize=14)
            st.pyplot(fig)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_heatmap.png")
            fig.savefig(path)
            plt.close(fig)
            st.session_state.plot_paths.append(path)

        if "Multi line trend" in st.session_state.get("selected_plots",[]) and len(numeric_cols) >= 2:
            st.markdown("#### 📈 Multi-Line Trend Plot")
            plt.figure(figsize=(10, 6))
            for col in numeric_cols:
                plt.plot(df[col], label=col)
            plt.legend()
            plt.title("Multi-Line Trend of Numeric Columns")
            st.pyplot(plt)
            path = os.path.join(st.session_state.temp_dir, f"table_{i}_line_trend.png")
            save_plt(path)

        cat_cols = [
            col for col in df.columns
            if df[col].dtype == 'object' or df[col].nunique() <= 5
        ]

        if "Categorical columns Plot" in st.session_state.get("selected_plots",[]) and cat_cols:
            st.markdown("#### 🪩 Categorical Columns Overview")
            for col in cat_cols:
                try:
                    plt.figure(figsize=(8, 5))
                    df[col].value_counts().plot(kind='bar', color='#FF6F61')
                    plt.title(f"Count Plot for '{col}'")
                    st.pyplot(plt)
                    path = os.path.join(st.session_state.temp_dir, f"table_{i}_cat_{safe_filename(col)}.png")
                    save_plt(path)
                except Exception as e:
                    st.warning(f"⚠️ Skipped column '{col}': {e}")

        else:
            st.info("No numeric data available for plotting in this table.")