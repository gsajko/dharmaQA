import os
from pathlib import Path

import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.vectorstores import LanceDB
from langchain_core.runnables import RunnablePassthrough

import lancedb

# HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm_repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=llm_repo_id, model_kwargs={"temperature": 0.1, "max_length": 256}
    )
    db_path = Path(".lancedb")
    db = lancedb.connect(db_path)
    table = db.open_table("dharma_qa")
    docsearch = LanceDB(table, embeddings)
    retriever = docsearch.as_retriever(search_kwargs={"k": 7})

    # Create system prompt
    template = """
    You are a respected spiritual teacher, Rob Burbea.
    Try to distill the following pieces of context to answer the question at the end.
    Question is asked by a student.
    If you don't know the answer, just say that you don't know.
    Don't try to make up an answer.
    Use five sentences maximum and keep the answer as concise as possible.
    Avoid answering questions that are not related to the dharma.
    If the question is not about the dharma,
    politely inform them that you are tuned to only answer
    questions about the dharma.

    {context}
    Question: {question}
    Helpful Answer:"""
    # Add system prompt to chain

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
