import streamlit as st
from vectorizer import *
from load_model import chat
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader

def process_file(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text_content = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()
            return text_content
            
        elif file_type == "text/plain":
            
            text_content = uploaded_file.read().decode("utf-8")
            return text_content
            


with st.sidebar:
    st.image("https://raw.githubusercontent.com/adelardw/images/refs/heads/main/brand_logo.png")
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/adelardw)"

st.title("📝 File Answering")

uploaded_file = st.file_uploader("Upload an article", type=("txt", "pdf"))

question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    
    content = process_file(uploaded_file)
    rag = RAGPipeline()
    vectorstore, retriever = rag.run(file_path_or_text=content)
    retrieved_docs = retriever.invoke(question)
    context = " ".join(doc.page_content for doc in retrieved_docs)
    output = chat(question=question, context=context)

    st.write("### Answer")
    st.write(output)
    







    
