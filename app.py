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
from dotenv import load_dotenv
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings


from data_analysis import extract_from_tables_pdf, analyze
from export import export_file
load_dotenv()

GOOGLE_API_KEY=st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)


#go through each and every pdf and extracts the texts from pdf
def get_pdf_text(pdf_docs):
    problems=[]
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        pdf.seek(0)         #file pointer at beginning
        for page in pdf_reader.pages:
            page_text=page.extract_text() or ""
            text+= page_text
            lines=text.splitlines()
            for line in lines:
                if "Q" in line or "Question" in line or "Problem" in line or "Write a function" in line or "solve" in line:
                    problems.append(line.strip())

    return text , problems

#go through each and every csv file and extracts the texts from csv
def get_csv_text(csv_docs):
    text=""
    for file in csv_docs:
        if file.name.endswith(".csv"):
            df=pd.read_csv(file)
            text+= df.to_string(index=False)        #converts df into texts
    return text



#dividing the text into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks



#storing chunks in the vector database and upload the embeddings
def get_vector_store(text_chunks):
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"})
    
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context , 
    and make sure to provide all the deatils , also you are a helpful AI assistant that can answer questions and generate code snippets when needed.
    Use the context to answer accurately. If it's a programming-related question, provide clean and a runnable code snippet.
    if the answer is not in
    provided context just say , "answer is not available in the context ", don't provide the wrong answer.
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


#wrt user
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
    st.set_page_config(page_title="InsightIQ", page_icon="📊", layout="wide")

    st.markdown(
        "<h1 style='text-align: center; color: ##c5fcc4;'>📊 InsightIQ</h1>"
        "<h4 style='text-align: center; color: grey;'>A Smart Assistant to Analyze & Chat with your Documents</h4><hr>",
        unsafe_allow_html=True
    )

    # Tabs for interaction
    tab1, tab2 = st.tabs(["💬", "📈"])

    # Session for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

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



    #tab1:chat wth pdf
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



    #tb2: analysis
    with tab2:
        st.subheader("📊 Extract and Analyze Tables from PDFs")
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