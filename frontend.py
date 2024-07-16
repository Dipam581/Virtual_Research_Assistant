import streamlit as st
from pdfDataExtracter import read_pdf, explain_query

st.set_page_config(
    page_title="VRA",
    page_icon="📖",
)
st.header("Welcome to Virtual Research Assistant! 👋")
st.sidebar.success("Select Your Assistant.")

if "pdf_response" not in st.session_state:
    st.session_state.pdf_response = []

def upload_PDF():
    pdf_file = st.file_uploader("Choose a PDF file", accept_multiple_files=False)
    if st.button("Upload PDF"):
        if pdf_file is not None:
            pdf_data = read_pdf(pdf_file)
            st.session_state.pdf_response.append(pdf_data)
            st.success("PDF uploaded successfully!")
        else:
            st.warning("Please upload a PDF file")

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
call = page_names_to_funcs[demo_name]()

query = st.chat_input("Type any queries regarding this content")
if query is not None:
    st.write(st.session_state.pdf_response[0])
    st.write(explain_query(query))

