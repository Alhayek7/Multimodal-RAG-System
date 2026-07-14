import os
from PIL import Image
import pytesseract

# تحديد مسار Tesseract (اختياري - إذا كان مثبتاً في مكان مختلف)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ===== دوال إنشاء المجلدات =====

def ensure_directories():
    """إنشاء المجلدات المطلوبة"""
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("storage/faiss_index", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    print("✅ Directories ensured")


# ===== دوال حفظ الملفات =====

def save_uploaded_file(file, upload_dir: str) -> str:
    """
    حفظ الملف المرفوع في المجلد المحدد
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    content = file.file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    if os.path.exists(file_path):
        print(f"✅ File saved: {file_path} (Size: {os.path.getsize(file_path)} bytes)")
    else:
        print(f"❌ Failed to save file: {file_path}")
    
    return file_path


def save_uploaded_image(file, upload_dir: str) -> str:
    """
    حفظ الصورة المرفوعة في المجلد المحدد
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    content = file.file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    if os.path.exists(file_path):
        print(f"🖼️ Image saved: {file_path} (Size: {os.path.getsize(file_path)} bytes)")
    else:
        print(f"❌ Failed to save image: {file_path}")
    
    return file_path


# ===== دوال معالجة الصور =====

def extract_text_from_image(image_path: str, language: str = "ara+eng") -> str:
    """
    استخراج النص من الصورة باستخدام Tesseract OCR
    
    Parameters:
    - image_path: مسار الصورة
    - language: لغة OCR (افتراضي: عربي + إنجليزي)
    
    Returns:
    - النص المستخرج من الصورة
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=language)
        text = text.strip()
        
        if text:
            print(f"✅ Text extracted: {len(text)} characters")
            print(f"📝 Preview: {text[:200]}...")
        else:
            print("⚠️ No text detected in image")
        
        return text
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return ""


def process_image_file(file_path: str, filename: str = None) -> dict:
    """
    معالجة ملف صورة: حفظ واستخراج النص
    
    Parameters:
    - file_path: مسار الصورة
    - filename: اسم الملف (اختياري)
    
    Returns:
    - قاموس يحتوي على: filename, content, source, source_type
    """
    if filename is None:
        filename = os.path.basename(file_path)
    
    text = extract_text_from_image(file_path)
    
    if not text:
        text = "[No text detected in image]"
    
    return {
        "filename": filename,
        "content": text,
        "source": file_path,
        "source_type": "image"
    }