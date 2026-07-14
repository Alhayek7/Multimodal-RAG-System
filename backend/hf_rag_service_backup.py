import os
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FAISS_INDEX_PATH,
    TOP_K,
)

# متغير عام لتخزين قاعدة المتجهات
vector_store = None

# تحميل نموذج التضمين (مجاني)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# تحميل نموذج التوليد (مجاني - صغير وسريع)
def get_llm():
    """تحميل نموذج توليد مجاني من HuggingFace"""
    model_name = "microsoft/DialoGPT-medium"  # نموذج صغير وسريع
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # إنشاء pipeline للتوليد
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )
    
    return HuggingFacePipeline(pipeline=pipe)


def load_pdf(file_path: str):
    """تحميل PDF"""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents):
    """تقسيم النص إلى قطع"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks):
    """بناء قاعدة متجهات باستخدام FAISS"""
    global vector_store
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store


def ingest_pdf(file_path: str) -> Dict[str, Any]:
    """معالجة PDF وبناء الفهرس"""
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    build_vector_store(chunks)
    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "message": "PDF indexed successfully",
    }


def get_retriever():
    """استرجاع المستندات"""
    global vector_store
    if vector_store is None:
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def build_rag_prompt(question: str, retrieved_docs):
    """بناء الـ Prompt مع السياق"""
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )
    return f"""You are a document-grounded assistant.
Answer the question using only the context below.
If the answer is not available in the context, say:
"I don't know based on the provided PDF."

Context:
{context}

Question:
{question}

Answer:"""


def ask_without_rag(question: str) -> str:
    """سؤال بدون RAG (إجابة عامة)"""
    llm = get_llm()
    prompt = f"Question: {question}\nAnswer:"
    response = llm.invoke(prompt)
    return response


def ask_with_rag(question: str) -> Dict[str, Any]:
    """سؤال مع RAG (إجابة مدعومة بالـ PDF)"""
    global vector_store
    
    if vector_store is None:
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )

    # استرجاع القطع الأكثر تشابهاً
    docs_and_scores = vector_store.similarity_search_with_score(question, k=TOP_K)
    retrieved_docs = [doc for doc, score in docs_and_scores]
    
    # بناء الـ Prompt
    prompt = build_rag_prompt(question, retrieved_docs)
    
    # توليد الإجابة
    llm = get_llm()
    response = llm.invoke(prompt)

    # تجهيز القطع المسترجعة
    chunks = []
    for doc, score in docs_and_scores:
        source = os.path.basename(doc.metadata.get("source", ""))
        page = doc.metadata.get("page", 0) + 1
        chunks.append(
            {
                "content": doc.page_content,
                "source": source,
                "page": page,
                "score": float(score),
            }
        )

    return {
        "answer": response,
        "chunks": chunks,
    }