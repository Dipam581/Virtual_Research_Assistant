from PyPDF2 import PdfReader
import re, os

from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from transformers import pipeline


class DocumentPage:
    def __init__(self, page_content):
        self.page_content = page_content

def chunkData(docs,chunk_size=200,chunk_overlap=30):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,length_function=len,is_separator_regex=True)
    doc = text_splitter.create_documents([docs])
    return doc

def embed_document(document):
    os.environ['PINECONE_API_KEY'] = "356688b7-fc9b-49ba-9c5f-7162954577cd"
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    response = (summarizer(document, max_length=130, min_length=30, do_sample=False))

    mod_document = ""
    for idx, item in enumerate(response):
        summary_text = item.get('summary_text', '')
        mod_document += summary_text
        if idx < len(response) - 1:
            mod_document += "\n\n"

    documents = chunkData(docs=mod_document)
    embedding = HuggingFaceEmbeddings()
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    index_name = "vradocs"

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

    return mod_document

corpus = []
def read_pdf(data):
    if data:

        reader = PdfReader(data)
        for page_number, page in enumerate(reader.pages):
            py_data = re.sub('[^a-zA-Z0-9]', " ", page.extract_text())
            corpus.append(py_data)

        return embed_document(corpus)


def explain_query(query):
    print("query: {}".format(query))
    print(corpus)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_hug = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.9, token="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"
    )
    chain = load_qa_chain(llm_hug, chain_type="stuff")

    def retrivequery(query, k=2):
        matching = index.similarity_search(query=query, k=k)
        return matching
    def retriveAnswer(query):
        doc_search = retrivequery(query)
        response = chain.run(input_documents=doc_search, question=query)
        return response
