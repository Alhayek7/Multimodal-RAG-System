import requests
import streamlit as st
import time

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Multimodal RAG System", layout="wide")

st.title("📚 Multimodal RAG System")
st.markdown("This app shows how Retrieval-Augmented Generation works using **PDFs** and **Images**.")

# -----------------------------
# Visual Flow
# -----------------------------
st.header("RAG Flow")
st.graphviz_chart(
    """
digraph {
    rankdir=LR;
    "PDF/Images" -> Chunking;
    Chunking -> Embeddings;
    Embeddings -> FAISS;
    Question -> Retriever;
    FAISS -> Retriever;
    Retriever -> "Top Chunks";
    "Top Chunks" -> Prompt;
    Question -> Prompt;
    Prompt -> LLM;
    LLM -> Answer;
}
"""
)

# -----------------------------
# Upload Section
# -----------------------------
st.header("1. Upload Files")

tab1, tab2 = st.tabs(["📄 PDF", "🖼️ Images"])

# ===== TAB 1: PDF =====
with tab1:
    st.subheader("Upload PDF File")
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader",
        label_visibility="collapsed"
    )

    if uploaded_pdf is not None:
        st.info(f"📄 File ready: {uploaded_pdf.name} ({uploaded_pdf.size} bytes)")
        
        if st.button("📤 Index PDF", key="index_pdf", type="primary"):
            with st.spinner("Uploading and processing PDF..."):
                try:
                    file_bytes = uploaded_pdf.getvalue()
                    files = {
                        "file": (uploaded_pdf.name, file_bytes, "application/pdf")
                    }
                    
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ PDF indexed successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📄 Pages", result.get("pages", 0))
                        with col2:
                            st.metric("📦 Chunks", result.get("chunks", 0))
                        
                        st.session_state.pdf_loaded = True
                        st.session_state.pdf_name = uploaded_pdf.name
                        
                    else:
                        st.error(f"❌ Failed to upload PDF: {response.status_code}")
                        st.write(f"Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend server. Please make sure the server is running.")
                    st.info("Run: `python -m uvicorn backend.main:app --reload`")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ===== TAB 2: Images =====
with tab2:
    st.subheader("Upload Images")
    uploaded_images = st.file_uploader(
        "Choose image files (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="image_uploader",
        label_visibility="collapsed"
    )

    if uploaded_images:
        for img in uploaded_images:
            st.info(f"🖼️ Image ready: {img.name} ({img.size} bytes)")
        
        if st.button("📤 Index Images", key="index_images", type="primary"):
            with st.spinner("Processing images..."):
                success_count = 0
                for img in uploaded_images:
                    try:
                        file_bytes = img.getvalue()
                        files = {
                            "file": (img.name, file_bytes, f"image/{img.type.split('/')[-1]}")
                        }
                        
                        response = requests.post(
                            f"{BACKEND_URL}/upload_image",
                            files=files,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            success_count += 1
                            st.success(f"✅ Image indexed: {img.name}")
                        else:
                            st.error(f"❌ Failed to index: {img.name}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend server.")
                        break
                    except Exception as e:
                        st.error(f"❌ Error processing {img.name}: {str(e)}")
                
                if success_count > 0:
                    st.session_state.images_loaded = True
                    st.session_state.images_count = success_count
                    st.success(f"✅ {success_count} image(s) indexed successfully!")

# -----------------------------
# Ask Question
# -----------------------------
st.header("2. Ask a Question")

question = st.text_input(
    "Question",
    placeholder="Ask something about the uploaded files...",
    key="question_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)

if ask_button:
    if not question:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question},
                    timeout=60
                )

                if response.status_code != 200:
                    st.error(f"❌ Failed to get answer: {response.status_code}")
                    st.write(response.text)
                else:
                    data = response.json()

                    st.header("3. Comparison: No RAG vs RAG")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("❌ Without RAG")
                        st.error("The model answers from general knowledge only.")
                        st.info(data.get("no_rag_answer", "No answer"))

                    with col2:
                        st.subheader("✅ With RAG")
                        st.success("The model answers using retrieved context from PDFs & Images.")
                        st.success(data.get("rag_answer", "No answer"))

                    # ===== عرض المقاطع المسترجعة =====
                    retrieved_chunks = data.get("retrieved_chunks", [])
                    if retrieved_chunks:
                        st.header("4. Retrieved Chunks")
                        
                        # عرض عدد المصادر
                        pdf_count = sum(1 for c in retrieved_chunks if c.get("source_type") == "pdf")
                        image_count = sum(1 for c in retrieved_chunks if c.get("source_type") == "image")
                        st.caption(f"📊 Sources: {pdf_count} from PDF, {image_count} from Images")
                        
                        for i, chunk in enumerate(retrieved_chunks, start=1):
                            source_type = chunk.get("source_type", "unknown")
                            icon = "📄" if source_type == "pdf" else "🖼️"
                            
                            with st.expander(
                                f"{icon} Chunk {i} | Source: {chunk.get('source', 'unknown')} | Type: {source_type} | Page: {chunk.get('page', 'N/A')}"
                            ):
                                st.markdown(
                                    f"""**Source:** `{chunk.get('source', 'unknown')}`  
**Type:** `{source_type}`
**Page:** `{chunk.get('page', 'N/A')}`
**Score:** `{chunk.get('score', 0):.4f}`
---
{chunk.get('content', 'No content')}"""
                                )
                    else:
                        st.info("ℹ️ No chunks retrieved. Please upload files first.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend server. Please make sure the server is running.")
                st.info("Run: `python -m uvicorn backend.main:app --reload`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# -----------------------------
# Sidebar - Status
# -----------------------------
with st.sidebar:
    st.header("📊 Status")
    
    # حالة الـ PDF
    if "pdf_loaded" in st.session_state and st.session_state.pdf_loaded:
        st.success(f"✅ PDF: {st.session_state.get('pdf_name', 'Unknown')}")
    else:
        st.warning("⚠️ No PDF loaded")
    
    # حالة الصور
    if "images_loaded" in st.session_state and st.session_state.images_loaded:
        st.success(f"✅ Images: {st.session_state.get('images_count', 0)} loaded")
    else:
        st.warning("⚠️ No images loaded")
    
    st.divider()
    
    # معلومات الخادم
    st.caption("🔗 Backend URL:")
    st.code(BACKEND_URL)
    
    # زر لإعادة تعيين الجلسة
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # عرض معلومات الصور من الخادم (اختياري)
    if st.button("📊 Show Server Images Info"):
        try:
            response = requests.get(f"{BACKEND_URL}/images", timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.info(f"Server has {data.get('count', 0)} images")
                for img in data.get("images", [])[:3]:
                    st.caption(f"🖼️ {img['filename']}")
            else:
                st.error("Could not fetch images info")
        except:
            st.error("Server not responding")