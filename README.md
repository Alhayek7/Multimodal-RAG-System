# RAG PDF Teaching Demo

Professional teaching/demo project for explaining how Retrieval-Augmented Generation (RAG) works with a single PDF.

- **Backend:** FastAPI + LangChain + FAISS
- **Frontend:** Streamlit
- **Models:** OpenAI embeddings + OpenAI chat model

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
6. [API Reference](#api-reference)
7. [Frontend Behavior](#frontend-behavior)
8. [Teaching Flow](#teaching-flow)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The system is split into two services:

1. **Backend (FastAPI)**
   - Accepts PDF uploads
   - Chunks and embeds the document
   - Stores vectors in FAISS
   - Answers questions both with and without retrieval
2. **Frontend (Streamlit)**
   - UI for upload + question asking
   - Visual comparison:
     - **Without RAG** (general model answer)
     - **With RAG** (grounded answer from retrieved chunks)
   - Displays retrieved chunks with source/page/similarity score

Data flow:

`PDF -> Chunking -> Embeddings -> FAISS -> Retrieval -> Prompt -> LLM -> Answer`

---

## Project Structure

```text
rag-pdf-teaching-demo/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── rag_service.py
│   ├── schemas.py
│   └── utils.py
├── frontend/
│   └── app.py
├── data/
│   └── uploads/
├── storage/
│   └── faiss_index/
├── .env
├── requirements.txt
└── README.md
```

---

## Configuration

Runtime settings are defined in `backend/config.py`:

- `CHUNK_SIZE = 300`
- `CHUNK_OVERLAP = 50`
- `TOP_K = 4`
- `EMBEDDING_MODEL = "text-embedding-3-small"`
- `LLM_MODEL = "gpt-4.1-mini"`
- `TEMPERATURE = 0`
- `UPLOAD_DIR = "data/uploads"`
- `FAISS_INDEX_PATH = "storage/faiss_index"`

Environment variable required in `.env`:

```env
OPENAI_API_KEY=your_real_openai_api_key
```

---

## Runtime and Dependency Versions

Use the following versions for a stable run.

### Python

- `Python 3.13.12`

### Core packages

- `fastapi==0.136.0`
- `uvicorn==0.45.0`
- `streamlit==1.57.0`
- `requests==2.32.5`
- `python-multipart==0.0.26`
- `python-dotenv==1.2.1`
- `langchain==1.2.15`
- `langchain-openai==1.1.16`
- `langchain-community==0.4.1`
- `langchain-text-splitters==1.1.2`
- `faiss-cpu==1.13.2`
- `pypdf==6.10.2`

### Verify installed versions (optional)

```powershell
python --version
pip show fastapi uvicorn streamlit requests python-multipart python-dotenv langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pypdf
```

---

## Quick Start

### 1) Install dependencies

```powershell
cd "D:\Workshop Material\Jupyter Notebooks\RAGApp\rag-pdf-teaching-demo"
pip install -r requirements.txt
```

### 2) Run backend (Terminal 1)

```powershell
cd "D:\Workshop Material\Jupyter Notebooks\RAGApp\rag-pdf-teaching-demo"
uvicorn backend.main:app --reload
```

Expected backend URL:

- `http://localhost:8000`

Health check:

- [http://localhost:8000/health](http://localhost:8000/health)

### 3) Run frontend (Terminal 2)

```powershell
cd "D:\Workshop Material\Jupyter Notebooks\RAGApp\rag-pdf-teaching-demo"
streamlit run frontend/app.py
```

Expected frontend URL:

- `http://localhost:8501`

---

## How the RAG Pipeline Works

### A) Ingestion (`POST /upload`)

1. Upload PDF from frontend.
2. Backend saves file in `data/uploads`.
3. `PyPDFLoader` reads pages into documents.
4. `RecursiveCharacterTextSplitter` creates chunks.
5. `OpenAIEmbeddings` converts chunks to vectors.
6. `FAISS.from_documents(...)` builds vector index.
7. Index is persisted under `storage/faiss_index`.

### B) Question Answering (`POST /ask`)

For each question, backend runs two paths:

1. **Without RAG:** send question directly to LLM.
2. **With RAG:**
   - run `similarity_search_with_score(question, k=TOP_K)`
   - take top chunks + their similarity scores
   - build grounded prompt from retrieved chunks
   - ask LLM to answer using only retrieved context

### Why similarity score is shown

Each retrieved chunk includes a score, so students can see ranking quality and understand:

**"The retriever ranks chunks by semantic similarity."**

---

## API Reference

### `GET /health`

Response:

```json
{"status":"ok"}
```

### `POST /upload`

Request:

- `multipart/form-data`
- field: `file` (PDF)

Response example:

```json
{
  "filename": "Employee-Handbook.pdf",
  "pages": 34,
  "chunks": 242,
  "message": "PDF indexed successfully"
}
```

### `POST /ask`

Request body:

```json
{"question":"What is the attendance policy?"}
```

Response fields:

- `question`
- `no_rag_answer`
- `rag_answer`
- `retrieved_chunks[]`:
  - `content`
  - `source`
  - `page`
  - `score`

---

## Frontend Behavior

The Streamlit app contains:

1. **RAG Flow diagram**
2. **Upload + index section**
3. **Comparison section**
   - Without RAG shown with `st.error(...)`
   - With RAG shown with `st.success(...)`
4. **Retrieved chunks section**
   - Displays source/page/content/`Similarity Score`

Backend URL used by frontend:

- `BACKEND_URL = "http://localhost:8000"` in `frontend/app.py`

---

## Teaching Flow

Recommended classroom order:

1. Upload PDF and build index.
2. Explain chunking and embeddings.
3. Explain FAISS retrieval and ranking.
4. Ask a question.
5. Compare no-RAG vs RAG answers.
6. Open retrieved chunks and inspect scores.
7. Emphasize grounding and traceability.

---

## Troubleshooting

- **Frontend cannot call backend**
  - verify backend is running on `http://localhost:8000`
  - verify `BACKEND_URL` in `frontend/app.py`

- **No useful retrieved chunks**
  - re-index after changing chunk settings
  - adjust `TOP_K` for broader context

- **Dependency issues on Windows**
  - activate your environment
  - run `pip install -r requirements.txt` again

---

## Core Message

The LLM does **not** read the PDF directly.  
FAISS retrieves the most semantically relevant chunks first.  
The LLM generates the answer using retrieved context.