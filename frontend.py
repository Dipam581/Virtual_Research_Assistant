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

def voice_assistant():
    prompt = st.chat_input("Say something")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")

    st.title("Echo Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    if prompt:
        st.session_state.messages.append({"role": "assistant", "content": response})

def chat_with_us():
    pass

page_names_to_funcs = {
    "Upload PDF": upload_PDF,
    "Voice Assistant": voice_assistant,
    "Chat With Us": chat_with_us,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
call = page_names_to_funcs[demo_name]()

if demo_name == "Upload PDF":
    query = st.chat_input("Type any queries regarding this content")
    if query is not None:
        st.write(st.session_state.pdf_response[0])
        with st.spinner("Query is being performed"):
            st.subheader(query)
            st.write(explain_query(query))

