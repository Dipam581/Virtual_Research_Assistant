from PyPDF2 import PdfReader
import re, os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import chromadb.utils.embedding_functions as embedding_functions
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from transformers import pipeline


class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def embed_document(document):
    os.environ['PINECONE_API_KEY'] = "356688b7-fc9b-49ba-9c5f-7162954577cd"
    pc = Pinecone(api_key="356688b7-fc9b-49ba-9c5f-7162954577cd")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # pc.create_index(
    #     name="VRA",
    #     dimension=2,  # Replace with your model dimensions
    #     metric="cosine",  # Replace with your model metric
    #     spec=ServerlessSpec(
    #         cloud="aws",
    #         region="us-east-1"
    #     )
    # )
    # index = PineconeVectorStore.from_documents(document, index_name="VRA", embedding=huggingface_ef)
    # huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     api_key="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",
    #     model_name="facebook/bart-large-cnn"
    # )
    print("start")
    response = (summarizer(document, max_length=130, min_length=30, do_sample=False))

    document = ""

    for idx, item in enumerate(response):
        summary_text = item.get('summary_text', '')
        document += summary_text
        if idx < len(response) - 1:
            document += "\n\n"
    return document




def read_pdf(data):
    # try:
    #     if data:
    #         corpus = []
    #         reader = PdfReader(data)
    #         for page in reader.pages:
    #             py_data = re.sub('[^a-zA-Z0-9]', " ", page.extract_text())
    #             embed_document(py_data)
    #             corpus.append(py_data)
    #
    # except:
    #     print("Error in reading documents")
    # finally:
    #     return
    if data:
        corpus = []
        reader = PdfReader(data)
        for page_number, page in enumerate(reader.pages):
            py_data = re.sub('[^a-zA-Z0-9]', " ", page.extract_text())
            corpus.append(py_data)

        return embed_document(corpus)
