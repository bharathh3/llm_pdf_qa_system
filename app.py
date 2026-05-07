import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="LearnBot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (BLACK + GREEN UI)
# -----------------------------
st.markdown("""
<style>

/* Entire App */
.stApp {
    background-color: #050816 !important;
    color: white !important;
}

/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #050816 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0b1020 !important;
}

/* Header */
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Toolbar */
[data-testid="stToolbar"] {
    right: 2rem;
}

/* Main block */
.block-container {
    padding-top: 2rem;
    color: white;
}

/* Text */
h1, h2, h3, h4, h5, h6, p, div, label, span {
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00ff66, #00cc55);
    color: black !important;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    padding: 0.7rem 1.2rem;
}

/* Input */
.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #00ff66 !important;
    border-radius: 10px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #0b1020 !important;
    border: 2px dashed #00ff66 !important;
    border-radius: 20px;
    padding: 20px;
}

/* Chat/message box */
.chat-box {
    background-color: #111827;
    border: 1px solid #00ff66;
    border-radius: 16px;
    padding: 20px;
    margin-top: 20px;
    color: white;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #00ff66;
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: #0b1020;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# GEMINI SETUP
# -----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# SIDEBAR HISTORY
# -----------------------------
st.sidebar.title("📜 History")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    st.sidebar.write("• " + item)

# -----------------------------
# MAIN UI
# -----------------------------
st.markdown('<div class="main-title">LearnBot</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Hello 👋 Welcome to LearnBot.<br>Upload your PDF to generate smart summaries and answers instantly.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("PDF Uploaded Successfully ✅")

    # Save PDF temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_db = Chroma.from_documents(
        texts,
        embeddings
    )

    # Combine PDF text
    full_text = "\n".join(
        [doc.page_content for doc in texts]
    )

    # Gemini Model
    model = genai.GenerativeModel(
        "models/gemini-2.5-flash"
    )

    # -----------------------------
    # AUTO SUMMARY
    # -----------------------------
    st.subheader("📘 Auto Summary")

    if st.button("⚡ Generate Summary"):

        summary_prompt = f"""
        Summarize this PDF clearly and professionally.

        PDF Content:
        {full_text[:12000]}
        """

        try:
            summary_response = model.generate_content(summary_prompt)

            st.markdown(
                f'<div class="chat-box">{summary_response.text}</div>',
                unsafe_allow_html=True
            )

            st.session_state.history.append(
                "Generated PDF Summary"
            )

        except Exception as e:
            st.error(f"Error: {e}")

    # -----------------------------
    # CHAT SECTION
    # -----------------------------
    st.subheader("🤖 Ask Questions")

    query = st.text_input(
        "Ask anything about the document..."
    )

    if query:

        docs = vector_db.similarity_search(query)

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
        Answer based only on the PDF context.

        Context:
        {context}

        Question:
        {query}
        """

        try:

            response = model.generate_content(prompt)

            st.markdown(
                f'<div class="chat-box">{response.text}</div>',
                unsafe_allow_html=True
            )

            st.session_state.history.append(query)

        except Exception as e:
            st.error(f"Error: {e}")