from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq  # بدلاً من langchain_google_genai
from backend.config import UPLOAD_DIR
from backend.schemas import AskRequest, AskResponse
from backend.utils import ensure_directories, save_uploaded_file, save_uploaded_image, process_image_file

# تحميل المتغيرات البيئية
load_dotenv()
def get_groq_response(question: str, context: str) -> str:
    """
    توليد إجابة من نموذج Groq بناءً على السياق المسترجع
    """
    if not context or context.strip() == "":
        return "⚠️ لا يوجد سياق كافٍ للإجابة على هذا السؤال. يرجى رفع ملفات PDF أو صور تحتوي على معلومات."

    try:
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "❌ مفتاح API مفقود. تأكد من وجود GROQ_API_KEY في ملف .env"

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=api_key
        )

        prompt_template = """
أنت مساعد ذكي ومتخصص في تحليل المستندات.
مهمتك هي الإجابة على الأسئلة بناءً على المعلومات المُقدمة لك في "السياق" فقط.
**قواعد مهمة:**
1. استخدم ONLY المعلومات من السياق للإجابة.
2. لا تستخدم معرفتك العامة أو معلومات من خارج السياق.
3. إذا كان الجواب غير موجود في السياق، اذكر ذلك بوضوح.
4. قدم إجابة منظمة وواضحة.

السياق:
{context}

السؤال:
{question}

الإجابة (بناءً على السياق فقط):
"""
        prompt = prompt_template.format(question=question, context=context)
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        print(f"❌ خطأ في Groq API: {e}")
        return f"❌ حدث خطأ أثناء توليد الإجابة: {str(e)}"
    
# ===== تعريف التطبيق =====
app = FastAPI(title="Multimodal RAG System (PDF + Images)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== متغيرات لتخزين المحتوى =====
pdf_content = ""
pdf_pages = 0
image_contents = []  # قائمة لتخزين النصوص المستخرجة من الصور

# ===== دوال البداية =====
@app.on_event("startup")
def startup():
    ensure_directories()

# ===== رفع PDF =====
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_content, pdf_pages
    
    print("="*50)
    print(f"📁 Received file: {file.filename}")
    print("="*50)
    
    # قراءة محتوى الملف
    try:
        content = await file.read()
        print(f"📊 Read {len(content)} bytes")
        
        # حفظ الملف
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"✅ File saved: {file_path}")
        print(f"📏 File size: {os.path.getsize(file_path)} bytes")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return {
            "filename": file.filename,
            "pages": 0,
            "chunks": 0,
            "message": f"Error saving file: {str(e)}"
        }
    
    # قراءة الـ PDF باستخدام pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pdf_pages = len(reader.pages)
        print(f"📄 Pages loaded: {pdf_pages}")
        
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "
        
        pdf_content = full_text.strip()
        print(f"✅ PDF loaded! Text length: {len(pdf_content)} characters")
        print(f"📝 First 200 chars: {pdf_content[:200]}")
        
    except Exception as e:
        pdf_content = ""
        pdf_pages = 0
        print(f"❌ ERROR reading PDF: {str(e)}")
        return {
            "filename": file.filename,
            "pages": 0,
            "chunks": 0,
            "message": f"Error reading PDF: {str(e)}"
        }
    
    print("="*50)
    
    return {
        "filename": file.filename,
        "pages": pdf_pages,
        "chunks": 10,
        "message": "PDF uploaded successfully" if pdf_pages > 0 else "PDF uploaded but could not read content"
    }


# ===== رفع الصور =====
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    global image_contents
    
    print("="*50)
    print(f"🖼️ Received image: {file.filename}")
    print("="*50)
    
    # حفظ الصورة
    try:
        file_path = save_uploaded_image(file, "data/images")
        
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return {
            "filename": file.filename,
            "message": f"Error saving image: {str(e)}"
        }
    
    # استخراج النص من الصورة
    try:
        result = process_image_file(file_path, file.filename)
        image_contents.append(result)
        print(f"✅ Image processed: {file.filename}")
        print(f"📝 Text length: {len(result['content'])} characters")
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        image_contents.append({
            "filename": file.filename,
            "content": f"[Error processing image: {str(e)}]",
            "source": file_path,
            "source_type": "image"
        })
    
    print("="*50)
    
    return {
        "filename": file.filename,
        "message": "Image processed successfully"
    }


# ===== عرض حالة الصور =====
@app.get("/images")
def get_images():
    """إرجاع قائمة بالصور المرفوعة ومحتواها"""
    return {
        "count": len(image_contents),
        "images": [
            {
                "filename": img["filename"],
                "content_preview": img["content"][:200] + "..." if len(img["content"]) > 200 else img["content"],
                "source": img["source"]
            }
            for img in image_contents
        ]
    }


# ===== السؤال والإجابة =====
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    global pdf_content, image_contents
    
    # ===== جمع كل المحتوى =====
    all_content = ""
    retrieved_chunks = []
    
    # 1. إضافة محتوى الـ PDF
    if pdf_content and pdf_content != "":
        all_content += f"PDF Content:\n{pdf_content}\n\n"
        retrieved_chunks.append({
            "content": pdf_content[:500],
            "source": "uploaded_file.pdf",
            "page": 1,
            "source_type": "pdf",  # ✅ تأكد من هذا
            "score": 0.95
        })
        print(f"✅ Added PDF chunk with source_type: pdf")  # للتأكد
    
    # 2. إضافة محتوى الصور
    for img in image_contents:
        if img["content"] and not img["content"].startswith("[Error"):
            all_content += f"Image ({img['filename']}) Content:\n{img['content']}\n\n"
            retrieved_chunks.append({
                "content": img["content"][:500],
                "source": img["filename"],
                "page": 0,
                "source_type": "image",  # ✅ تأكد من هذا
                "score": 0.90
            })
            print(f"✅ Added image chunk with source_type: image")  # للتأكد
    
    # ===== إجابة بدون RAG =====
    no_rag_answer = f"سؤال عام: '{request.question}'. هذه إجابة من المعرفة العامة (بدون استخدام الملفات المرفوعة)."
    
    # ===== إجابة مع RAG =====
    if all_content:
        rag_answer = get_groq_response(request.question, all_content)
    else:
        rag_answer = "⚠️ لم يتم رفع أي PDF أو صور. يرجى رفع ملفات أولاً."
    
    # ===== إرجاع النتيجة =====
    return {
        "question": request.question,
        "no_rag_answer": no_rag_answer,
        "rag_answer": rag_answer,
        "retrieved_chunks": retrieved_chunks,
    }


# ===== حالة الخادم =====
@app.get("/health")
def health():
    return {
        "status": "ok",
        "pdf_loaded": bool(pdf_content),
        "images_count": len(image_contents)
    }