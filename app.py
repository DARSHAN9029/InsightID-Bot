import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
import torch


st.set_page_config(page_title="InsightIQ", page_icon="📊", layout="wide")

from data_analysis import extract_from_tables_pdf, analyze
from export import export_file

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


#PDF and CSV
def get_pdf_text(pdf_docs):
    problems=[]
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        pdf.seek(0)         
        for page in pdf_reader.pages:
            page_text=page.extract_text() or ""
            text+= page_text
            lines=text.splitlines()
            for line in lines:
                if "Q" in line or "Question" in line or "Problem" in line or "Write a function" in line or "solve" in line:
                    problems.append(line.strip())
    return text , problems


def get_csv_text(csv_docs):
    text=""
    for file in csv_docs:
        if file.name.endswith(".csv"):
            df=pd.read_csv(file)
            text+= df.to_string(index=False)        
    return text


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":device})
    
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template="""
    You are an intelligent and reliable AI assistant named "InsightIQ - A bot who can analyze and chat with the documents" created by Darshan Jain , designed to extract meaningful insights from the document attached
    and generate accurate responses from the provided context with respect to the document.
    You can also respond politely to general questions like your identity and greetings. (e.g: "Hello" ,"Who are you?" , etc). 
    Your responsibilities include answering questions in detail, generating correct , clean and executable code if needed and asked , and staying strictly within the information provided.

    INSTRUCTIONS:
    - Always use only the context provided below to answer the question.
    - If the question is about content, facts, or data, always use only the provided context to answer.
    - If the question is about code, provide clean, well-commented, and runnable code snippets.
    - If the answer is not found in the context, respond with:
    "The answer is not available in the provided context."
    - Do NOT generate answers based on your own knowledge if it’s not present in the context.

    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-2.5-pro",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


def user_input(user_question):
    embbeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db=FAISS.load_local("faiss_index",embbeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True)
    
    print(response)
    return response["output_text"]


#streamlit app
def main():
    st.markdown(
        "<h1 style='text-align: center; color: ##c5fcc4;'>📊 InsightIQ</h1>"
        "<h4 style='text-align: center; color: grey;'>A Smart Assistant to Analyze & Chat with your Documents</h4><hr>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["💬 CHAT", "📈 ANALYZE"])

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_tabular" not in st.session_state:
        st.session_state.is_tabular=False
    if "analysis_triggered" not in st.session_state:
        st.session_state.analysis_triggered = False
    if "plots_generated" not in st.session_state:
        st.session_state.plots_generated= False

    with st.sidebar:
        st.title("📁 Upload Zone")
        pdf_docs = st.file_uploader("Upload PDF or CSV files", type=["pdf", "csv"], accept_multiple_files=True)

        if st.button("🚀 Submit"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF or CSV file.")
            else:
                with st.spinner("Processing your files..."):
                    raw_text = ""
                    for file in pdf_docs:
                        if file.name.endswith(".pdf"):
                            pdf_text , questions= get_pdf_text([file])
                            raw_text += pdf_text
                        elif file.name.endswith(".csv"):
                            raw_text += get_csv_text([file])

                    if raw_text.strip() == "":
                        st.error("No readable content found in uploaded files.")
                    else:
                        chunks = get_text_chunks(raw_text)
                        get_vector_store(chunks)
                        st.success("Files processed and embedded successfully!")
                        st.toast("Uploaded and Vectorized!", icon="✅")
    # Chat Tab
    with tab1:
        if st.session_state.get("messages"):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_query = st.chat_input("Ask a question about your uploaded files...")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            with st.spinner("InsightIQ says..."):
                try:
                    reply = user_input(user_query)
                except Exception as e:
                    reply = f"❌ Error: {e}"

            with st.chat_message("assistant"):
                st.markdown(reply)

            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": reply})


    # Analysis Tab
    with tab2:
        st.markdown("### 🎯 Select Plots to Generate")      
        plot_options=st.multiselect(
            "Choose plots you want to include in analysis:",
            ["Scatter Plot" , "Bar Plot" , "Scatter Matrix" , "Co-relation Heatmap" , "Multi line trend" , "Categorical columns Plot"],
            default=["Scatter Plot" , "Bar Plot"]
        )
        st.session_state["selected_plots"]=plot_options
        
        if st.button("🔍 Run Table Analysis"):
            if not pdf_docs:
                st.warning("Please upload a PDF file for analysis.")
            else:
                with st.spinner("Extracting tables and analyzing..."):
                    tables = extract_from_tables_pdf(pdf_docs)
                    st.session_state.tables = tables
                    analyze(tables)
                    st.success("Table extraction and analysis complete!")

        if st.button("📄 Export PDF"):      
            if "tables" not in st.session_state:
                st.warning("Please analyze the document first")
            else:
                with st.spinner("Generating PDF..."):
                    pdf_path = export_file(st.session_state.tables)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=f,
                            file_name="insightiq_report.pdf",
                            mime="application/pdf"
                        )


if __name__== "__main__":
    main()