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
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#go through each and every pdf and extracts the texts from pdf
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        pdf.seek(0)         #file pointer at beginning
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


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
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context , and make sure to provide all the deatils , if the answer is not in
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
    st.write("REPLY: " , response["output_text"])


#streamlit app
def main():
    st.set_page_config("Chat with multiple pdf")
    st.header("InsightIQ: A bot who can Analyze and Chat!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs=st.file_uploader("Upload your pdf file and click on the Submit button",accept_multiple_files=True,type=["pdf","csv"])
        if st.button("Submit"):
            with st.spinner("Processing your files...."):

                raw_text=""

                for file in pdf_docs:
                    if file.name.endswith(".pdf"):
                        raw_text+=get_pdf_text([file])
                    elif file.name.endswith(".csv"):
                        raw_text+=get_csv_text([file])

                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

    if st.button("Analysis"):
        with st.spinner("Extracting and analyzing tables..."):
            tables = extract_from_tables_pdf(pdf_docs)
            analyze(tables)

if __name__== "__main__":
    main()