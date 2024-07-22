import os
import re
from PyPDF2 import PdfReader
from transformers import pipeline
import streamlit as st

from database import insert_doc

from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


class DocumentPage:
    def __init__(self, page_content):
        self.page_content = page_content


def chunkData(docs, chunk_size=1000, chunk_overlap=200):
    if isinstance(docs, list):
        docs = " ".join(docs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=True
    )
    doc_chunks = text_splitter.create_documents([docs])
    return doc_chunks


def retrivequery(query, k=2):
    documents = chunkData(docs=corpus)
    embedding = HuggingFaceEmbeddings()
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    index_name = "first"
    index = PineconeVectorStore.from_documents(documents, index_name=index_name, embedding=embedding)
    matching = index.similarity_search(query=query, k=k)
    return matching


indexes = {
    "first": "first",
    "second": "second",
    "third": "third"
}


def embed_document(document):
    os.environ['PINECONE_API_KEY'] = "356688b7-fc9b-49ba-9c5f-7162954577cd"
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarizer(document, max_length=200, min_length=50)

    mod_document = ""
    for idx, item in enumerate(summary):
        summary_text = item.get('summary_text', '')
        mod_document += summary_text
        if idx < len(summary) - 1:
            mod_document += "\n\n"

    documents = chunkData(docs=mod_document)
    print("End of chunking of data")
    embedding = HuggingFaceEmbeddings()
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    index_name = "first"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = PineconeVectorStore.from_documents(documents, index_name=index_name, embedding=embedding)

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_hug = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.6, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    chain = load_qa_chain(llm_hug, chain_type="stuff")

    def retriveAnswer(query):
        doc_search = retrivequery(query)
        response = chain.run(input_documents=doc_search, question=query)
        return response

    answer = retriveAnswer("Create a one liner Title for this document")

    send_data = {
        "name": answer,
        "index_name": index_name,
        "index": index
    }
    status = insert_doc(send_data)
    if status:
        print("status: ", status)
        st.sidebar.header(answer)
        st.write(mod_document)
        return mod_document
    else:
        return False


corpus = []


def read_pdf(data):
    print("enter in script")
    if data:
        reader = PdfReader(data)
        for page_number, page in enumerate(reader.pages):
            page_data = re.sub('[^a-zA-Z0-9]', " ", page.extract_text())
            corpus.append(page_data)
        return embed_document(" ".join(corpus))


def explain_query(query):
    print("query: {}".format(query))
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_hug = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.9, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    chain = load_qa_chain(llm_hug, chain_type="stuff")

    def retriveAnswer(query):
        doc_search = retrivequery(query)
        response = chain.run(input_documents=doc_search, question=query)
        return response

    answer = retriveAnswer(query)
    return answer
