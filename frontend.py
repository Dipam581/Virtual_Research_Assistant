import streamlit as st
import time
from pdfDataExtracter import read_pdf, explain_query

st.set_page_config(
    page_title="VRA",
    page_icon="📖",
)
st.header("Welcome to Virtual Research Assistant! 👋")
st.sidebar.success("Select Your Assistant.")

pdf_response = ""
def upload_PDF():
    pdf_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False)
    with st.spinner('Data is processing...'):
        pdf_response = read_pdf(pdf_file)
        st.write(pdf_response)


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

if pdf_response:
    query = st.text_input("Type any queries regarding this content")
    if st.button("Explain"):
        if query:
            explain_query(query)
