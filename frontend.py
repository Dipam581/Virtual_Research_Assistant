import streamlit as st
import time
from pdfDataExtracter import read_pdf

st.set_page_config(
    page_title="VRA",
    page_icon="📖",
)
st.header("Welcome to Virtual Research Assistant! 👋")
st.sidebar.success("Select a service.")


def upload_PDF():
    pdf_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False)
    st.spinner('Data is processing...')
    response = read_pdf(pdf_file)
    st.write(response)


def upload_DOC():
    pass

def chat_with_us():
    pass

page_names_to_funcs = {
    "Upload PDF": upload_PDF,
    "Upload DOCs": upload_DOC,
    "Chat With Us": chat_with_us,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
