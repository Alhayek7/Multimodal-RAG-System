from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os

from backend.config import UPLOAD_DIR
from backend.schemas import AskRequest, AskResponse
from backend.utils import ensure_directories, save_uploaded_file

app = FastAPI(title="RAG PDF Teaching Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# متغير لتخزين محتوى PDF للاختبار
pdf_content = ""

@app.on_event("startup")
def startup():
    ensure_directories()


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global pdf_content
    file_path = save_uploaded_file(file, UPLOAD_DIR)
    
    # قراءة محتوى الـ PDF للتجربة
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        pdf_content = " ".join([doc.page_content for doc in documents[:5]])  # أول 5 صفحات
        pages = len(documents)
    except Exception as e:
        pdf_content = f"Error reading PDF: {str(e)}"
        pages = 0
    
    return {
        "filename": file.filename,
        "pages": pages,
        "chunks": 10,
        "message": "PDF uploaded successfully"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    global pdf_content
    
    # إجابة بدون RAG (من المعرفة العامة)
    no_rag_answer = f"هذا سؤال عام: '{request.question}'. ليس لدي سياق محدد للإجابة."
    
    # إجابة مع RAG (من الـ PDF)
    if pdf_content:
        rag_answer = f"بناءً على محتوى الـ PDF: {pdf_content[:500]}..."
    else:
        rag_answer = "لم يتم رفع أي PDF بعد. يرجى رفع ملف PDF أولاً."
    
    # قطع مسترجعة (محاكاة)
    retrieved_chunks = [
        {
            "content": pdf_content[:200] if pdf_content else "No content available",
            "source": "sample.pdf",
            "page": 1,
            "score": 0.95
        }
    ]
    
    return {
        "question": request.question,
        "no_rag_answer": no_rag_answer,
        "rag_answer": rag_answer,
        "retrieved_chunks": retrieved_chunks,
    }


@app.get("/health")
def health():
    return {"status": "ok"}