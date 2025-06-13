from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def load_and_embed_pdfs(pdf_folder_path):
    all_docs = []
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder_path, filename))
            docs = loader.load()
            all_docs.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    batch_size = 100
    for i in tqdm(range(0, len(split_docs), batch_size)):
        batch_docs = split_docs[i:i+batch_size]
        try:
            vectordb.add_documents(batch_docs)
        except Exception as e:
            print(f"❌ Batch {i} 실패: {e}")
            continue

    vectordb.persist()
    return vectordb

def get_qa_chain():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, max_tokens=1024),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def get_answer(question, qa_chain):
    result = qa_chain({"query": question})
    if not result["source_documents"]:
        return "죄송합니다. 관련 문서를 찾지 못했습니다."
    return result["result"]
