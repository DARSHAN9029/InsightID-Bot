# 📊InsightIQ : A bot who can Analyze and Chat!
A powerful Streamlit-based web application by using "Gemini 1.5 Flash" model and "HuggingFaceEmbeddings" that lets you:

- 📥 Upload PDF or CSV files(Click "*Browse*" for uploading the file)
- 💬 Chat with content from PDFs (text-based)
- 📈 Automatically analyze and visualize tabular data
- 🔍 Discover trends, patterns, and relationships across columns

---
## 🔗Live App
[Click here to visit deployed Streamlit app](https://insightiq-bot.streamlit.app/)

---
## 🚀 Features

### 🗂 File Upload Support
- Upload multiple .pdf or .csv files
- Automatically detects and extracts(Click "*Submit*" for processing of file):
  - Text content from PDFs (for Q&A/chat)
  - Tables from PDFs or CSVs (for data analysis)

### 🧠 PDF Chatbot (Text Mode)
- Extracts and stores page-wise PDF text
- Integrates with a GenAI model to allow users to ask questions based on the content
- Ideal for querying reports, contracts, or scanned academic papers

### 📊 Automatic Data Analysis
Supports any structured table inside PDFs or CSV files(Click "*Analyze*" for analyzing the file containing any tabular data).

🔹 Visualizations include:
- Scatter Plot of first 2 numeric columns
- Bar Chart of most variable column
- Scatter Matrix (pairwise comparison of numeric columns)
- Correlation Heatmap
- Multi-Line Trend Plot for time/index-based data

🔹 Handles:
- Numeric column detection
- Type conversion
- Plot rendering without button clicks

---
## 🛠 Installation

### 1. Clone the repo
bash
git clone https://github.com/DARSHAN9029/InsightID-Bot.git

### 2. Create a virtual environment

python -m venv myenv

### 3. Install dependencies

pip install -r requirements.txt

---
▶ Running the App
bash
streamlit run app.py

---
## Screehshots:
# example of a car data(cleaned_car.csv):
![Screenshot 2025-06-13 004625](https://github.com/user-attachments/assets/6cefc32a-4703-41a3-a893-8795b2397f05)
![Screenshot 2025-06-13 004641](https://github.com/user-attachments/assets/fa5169e1-97a9-4ca7-acfd-eaad0e7feaa7)
![Screenshot 2025-06-13 004700](https://github.com/user-attachments/assets/7c89a8a5-f170-4396-b0ae-5126184b9cb9)
![Screenshot 2025-06-13 004718](https://github.com/user-attachments/assets/fb99882b-0150-4104-8029-ee067423cae7)
