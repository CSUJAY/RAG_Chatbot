import os
import fitz
import docx
import re
import psutil
import json
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
import tempfile

# -------------------- CONFIG --------------------
CHUNK_LINES = 20
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "doc_chunks"
# ------------------------------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_qdrant():
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    return client

def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        lines = page.get_text().splitlines()
        for i in range(0, len(lines), CHUNK_LINES):
            chunk_lines = lines[i:i + CHUNK_LINES]
            chunk_text = "\n".join(chunk_lines)
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "filename": Path(pdf_path).name,
                    "page": page_num,
                    "chunk_id": f"{page_num}-{i//CHUNK_LINES}",
                    "line_range": f"{i+1}-{i+len(chunk_lines)}"
                }
            })
    return chunks

def extract_chunks_from_docx(docx_path):
    doc = docx.Document(docx_path)
    lines = [para.text for para in doc.paragraphs if para.text.strip()]
    chunks = []
    for i in range(0, len(lines), CHUNK_LINES):
        chunk_lines = lines[i:i + CHUNK_LINES]
        chunk_text = "\n".join(chunk_lines)
        chunks.append({
            "content": chunk_text,
            "metadata": {
                "filename": Path(docx_path).name,
                "page": 1,
                "chunk_id": f"1-{i//CHUNK_LINES}",
                "line_range": f"{i+1}-{i+len(chunk_lines)}"
            }
        })
    return chunks

def index_documents(files, embedder, qdrant):
    all_chunks = []
    for file in files:
        suffix = Path(file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        if suffix == ".pdf":
            all_chunks.extend(extract_chunks_from_pdf(tmp_path))
        elif suffix == ".docx":
            all_chunks.extend(extract_chunks_from_docx(tmp_path))
    vectors = embedder.encode([c["content"] for c in all_chunks])
    points = [
        PointStruct(id=i, vector=vectors[i], payload=all_chunks[i]["metadata"] | {"text": all_chunks[i]["content"]})
        for i in range(len(all_chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

def retrieve_top_chunks(query, embedder, qdrant, top_k=3, filter_filename=None):
    query_vec = embedder.encode(query).tolist()
    search_filter = None
    if filter_filename:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filter_filename)
                )
            ]
        )
    hits = qdrant.search(
        COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        query_filter=search_filter
    )
    return hits

def highlight_relevant_line(chunk, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    lines = chunk.splitlines()
    highlighted = [f"👉 **{line}**" if pattern.search(line) else line for line in lines]
    return "\n".join(highlighted)

def show_resource_usage():
    mem = psutil.virtual_memory()
    st.sidebar.markdown(f"**RAM Usage:** {mem.percent}%")
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            st.sidebar.markdown(f"**GPU {gpu.id}:** {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
    except:
        st.sidebar.markdown("GPU info not available (install `GPUtil`)")

# -------------------- UI --------------------
st.set_page_config(page_title="Advanced Doc QA", layout="wide")
st.title("📚 Advanced Offline QA Bot")

show_resource_usage()

uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    filenames = [f.name for f in uploaded_files]
    selected_file = st.selectbox("Filter by document (optional)", ["All"] + filenames)

query = st.text_input("Ask a question about the uploaded documents:")

if uploaded_files and query:
    with st.spinner("Processing..."):
        embedder = load_embedder()
        qdrant = load_qdrant()

        if not qdrant.count(COLLECTION_NAME).count:
            index_documents(uploaded_files, embedder, qdrant)

        filter_name = selected_file if selected_file != "All" else None
        hits = retrieve_top_chunks(query, embedder, qdrant, top_k=3, filter_filename=filter_name)

        if hits:
            st.markdown("### 🔍 Top Matching Chunks")
            best = hits[0].payload
            for i, hit in enumerate(hits, 1):
                context = hit.payload
                score = hit.score
                highlighted = highlight_relevant_line(context["text"], query)
                with st.expander(f"Match #{i} — Score: {score:.3f}"):
                    st.markdown(highlighted)
                    st.markdown(f"""
**Source:**  
- 📄 File: `{context['filename']}`  
- 📄 Page: `{context['page']}`  
- 🔖 Chunk ID: `{context['chunk_id']}`  
- 📌 Line Range: `{context['line_range']}`
""")
            st.download_button("📥 Download Best Answer", json.dumps(best, indent=2), file_name="best_answer.json")
        else:
            st.warning("Answer not found in the provided document context.")