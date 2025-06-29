---

# 📚 Advanced Offline QA Bot

An offline question-answering app built with Streamlit, Qdrant, and SentenceTransformers. It supports uploading `.pdf` and `.docx` documents and querying them for relevant information using semantic search.

---

## 🚀 Features

* Upload and index PDF or DOCX files
* Chunk documents into manageable sections
* Embed text using SentenceTransformers
* Store and query embeddings using Qdrant (in-memory)
* Filter results by file
* Highlight matching content and download best match
* View real-time system resource usage

---

## 🛠️ Setup Instructions

1. Clone the repo

   ```bash
   git clone https://github.com/yourusername/advanced-doc-qa.git
   cd advanced-doc-qa
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app

   ```bash
   streamlit run app.py
   ```

---

📁 File Structure

```
.
├── app.py              # Main Streamlit app
├── requirements.txt    # All dependencies
└── README.md
```

---

🧠 Architectural Overview

* **Frontend:** Streamlit UI for document upload, querying, filtering, and displaying results.
* **Embedding Layer:** Uses `sentence-transformers/all-MiniLM-L6-v2` for generating 384-dimensional embeddings.
* **Vector Store:** Qdrant is used in in-memory mode to store and retrieve chunks via cosine similarity.
* **Chunking Engine:** Breaks documents into fixed-sized line chunks with rich metadata.
* **System Monitoring:** Uses `psutil` (and optionally `GPUtil`) to report resource usage in the sidebar.

---

✂️ Chunking Strategy

* Documents are split into **chunks of 20 lines** (configurable via `CHUNK_LINES`).
* Each chunk carries metadata:

  * `filename`, `page`, `chunk_id`, and `line_range`
* This balance reduces context loss while keeping embedding sizes manageable.

---

🔍 Retrieval Approach

1. The user query is embedded using the same SentenceTransformer.
2. A **semantic similarity search** (cosine distance) is performed against stored vectors using Qdrant.
3. Optionally, results can be **filtered by filename**.
4. Top-k matching chunks (default: 3) are returned with metadata and relevance score.
5. Matching lines within chunks are **highlighted** using regex.

---

💻 Hardware Usage

* **RAM:** Displayed live using `psutil`. Useful for monitoring large document loads.
* **GPU (optional):** If available and `GPUtil` is installed, GPU memory usage is shown.
* **Qdrant:** Runs in memory (`:memory:`) for simplicity; no external DB setup needed.

---

🧪 Observations & Notes

* Performance is fast for small- to medium-sized documents.
* Using in-memory Qdrant is suitable for lightweight QA apps; for production use, connect to a persistent Qdrant instance.
* The `all-MiniLM-L6-v2` model provides a strong balance of speed and accuracy.
* Current chunking strategy is line-based; for longer documents or paragraphs, token-based or sentence-based chunking might yield better semantic granularity.

---

✅ To Do / Future Improvements

* Persistent Qdrant storage backend
* Support for `.txt`, `.md`, and other file types
* Improve chunk overlap logic
* UI enhancements (pagination, file manager)
* Add unit tests and CI

---



Let me know if you'd like a `requirements.txt` or an icon/logo to go along with this!
