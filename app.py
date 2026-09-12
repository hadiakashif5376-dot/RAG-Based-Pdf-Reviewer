import os
import tempfile

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load environment variables locally.
# On Streamlit Cloud, add GROQ_API_KEY in App Settings > Secrets.
load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF and ask questions using Retrieval-Augmented Generation (RAG).")


# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

# Keep chunks reasonably small for retrieval.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load the open-source embedding model once."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    """Create the Groq client once."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# PDF processing
# -----------------------------
def extract_pdf_text(uploaded_file):
    """Extract text from all pages of the uploaded PDF."""
    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages)


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character-based chunks.

    Note:
    The embedding model tokenizes each chunk internally before
    creating its vector embedding.
    """
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(chunks, embedding_model):
    """Create embeddings and store them in an open-source FAISS index."""
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    # Inner Product + normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve_chunks(question, chunks, index, embedding_model, top_k=TOP_K):
    """Retrieve the most relevant chunks for a user question."""
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_embedding = np.asarray(question_embedding, dtype="float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append(
                {
                    "text": chunks[idx],
                    "score": float(score),
                }
            )

    return results


# -----------------------------
# Groq generation
# -----------------------------
def generate_answer(question, retrieved_chunks, groq_client):
    """Send retrieved context + question to the Groq-hosted model."""
    context = "\n\n".join(
        [
            f"[Source {i + 1}]\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not invent information.
2. If the answer is not present in the context, clearly say:
   "I could not find the answer in the uploaded document."
3. Give a clear and concise answer.
4. When possible, mention which retrieved source supports the answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You answer questions from retrieved document context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=GROQ_MODEL,
        temperature=0.2,
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ RAG Settings")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write("**Vector DB:** FAISS")
    st.write("**Frontend:** Streamlit")
    st.write("**LLM API:** Groq")


# -----------------------------
# API key check
# -----------------------------
groq_client = load_groq_client()

if not groq_client:
    st.warning(
        "GROQ_API_KEY is not configured. "
        "Add it to your local .env file or Streamlit Cloud Secrets."
    )


# -----------------------------
# Upload and process PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    if st.session_state.document_name != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            text = extract_pdf_text(uploaded_file)

            if not text.strip():
                st.error(
                    "No extractable text was found. "
                    "This app currently works with text-based PDFs, "
                    "not scanned image-only PDFs."
                )
                st.stop()

            chunks = create_chunks(text)

            if not chunks:
                st.error("Could not create text chunks from this PDF.")
                st.stop()

            embedding_model = load_embedding_model()
            index = create_faiss_index(chunks, embedding_model)

            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
            st.session_state.document_name = uploaded_file.name

        st.success(
            f"PDF processed successfully: {len(st.session_state.chunks)} chunks created."
        )

# -----------------------------
# Question answering
# -----------------------------
if st.session_state.faiss_index is not None:
    st.divider()

    st.subheader("💬 Ask a question")

    question = st.text_input(
        "What would you like to know about the document?",
        placeholder="Example: What are the main objectives of this document?",
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not groq_client:
            st.error("GROQ_API_KEY is missing.")
        else:
            with st.spinner("Searching the document and generating an answer..."):
                embedding_model = load_embedding_model()

                retrieved = retrieve_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.faiss_index,
                    embedding_model,
                )

                answer = generate_answer(
                    question,
                    retrieved,
                    groq_client,
                )

            st.subheader("Answer")
            st.write(answer)

            with st.expander("🔎 Retrieved context"):
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Source {i} — similarity: {item['score']:.3f}**"
                    )
                    st.write(item["text"])
                    st.divider()
else:
    st.info("Upload a PDF to build the RAG knowledge base.")
