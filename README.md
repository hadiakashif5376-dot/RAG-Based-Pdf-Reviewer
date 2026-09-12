# 📚 PDF RAG Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF document and ask questions about its content.

The application extracts text from the PDF, splits it into chunks, creates vector embeddings using an open-source embedding model, stores the embeddings in **FAISS**, retrieves the most relevant chunks for a question, and uses an open-source LLM through **Groq API** to generate the final answer.

---

## 🚀 Features

* Upload a PDF document
* Extract text from the PDF
* Split document text into overlapping chunks
* Generate embeddings using an open-source embedding model
* Store embeddings in FAISS vector database
* Perform similarity search
* Retrieve the most relevant document chunks
* Ask questions about the uploaded document
* Generate answers using `openai/gpt-oss-120b` through Groq
* Display the retrieved context used to generate the answer
* Simple Streamlit interface
* Ready for GitHub and Streamlit Cloud deployment

---

## 🧠 How RAG Works

The application follows this workflow:

```text
                 PDF Upload
                     ↓
             Extract PDF Text
                     ↓
               Text Chunking
                     ↓
            Embedding Model
                     ↓
             Vector Embeddings
                     ↓
                  FAISS
              Vector Database
                     ↓
              User Question
                     ↓
          Question Embedding
                     ↓
           FAISS Similarity Search
                     ↓
          Top Rel
```
