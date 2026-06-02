<h1 align="center">Chat with PDF</h1>

<p align="center">
  <b>Upload a PDF. Ask anything. Get answers grounded in the actual document — with source citations.</b>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C.svg" alt="LangChain"></a>
  <a href="#"><img src="https://img.shields.io/badge/LLM-Gemini%202.5-4285F4.svg" alt="Gemini 2.5"></a>
  <a href="#"><img src="https://img.shields.io/badge/Vector%20DB-FAISS-009688.svg" alt="FAISS"></a>
  <a href="#"><img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg" alt="Streamlit"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

---

## Overview

A vanilla **Retrieval-Augmented Generation (RAG)** application that turns any PDF
into a queryable knowledge base. The document is split into overlapping chunks,
embedded with the open-source `BAAI/bge-small-en` model, and indexed in **FAISS**.
At query time, the user's question is embedded, the top-K most similar chunks
are retrieved, and **Google Gemini 2.5** generates a grounded answer with
inline source citations — chunk ID and page number — so nothing is hallucinated.

Built end-to-end in Python with **LangChain** and a **Streamlit** chat UI.

---

## Demo

> Add a GIF or screenshot here once you record one. Suggested filename: `docs/demo.gif`.

```
![demo](docs/demo.gif)
```

---

## Architecture

```
                ┌──────────────────────  INGEST  ──────────────────────┐
                │                                                      │
   PDF ──► PyPDFLoader ──► Chunker ──► BGE-Small Embeddings ──► FAISS  │
                │         (1000/200)        (384-dim)         (on-disk)│
                └──────────────────────────────────────────────────────┘
                                                                │
                                                                ▼
                ┌──────────────────────  QUERY  ───────────────────────┐
                │                                                      │
   Question ──► Embedding ──► Similarity Search ──► Top-K Chunks       │
                                                          │            │
                                                          ▼            │
                                              Gemini 2.5 + Prompt      │
                                                          │            │
                                                          ▼            │
                                         Answer + Source Citations     │
                └──────────────────────────────────────────────────────┘
```

This is **vanilla RAG** — one retrieval, one LLM call. Agentic RAG, which can
plan multiple retrievals and use external tools, is a planned extension.

---

## Tech Stack

| Layer            | Choice                              | Why                                                 |
|------------------|-------------------------------------|-----------------------------------------------------|
| Frontend         | Streamlit                           | Zero-boilerplate Python UI                          |
| Backend          | Python 3.10+                        | Standard for the ML/AI ecosystem                    |
| RAG framework    | LangChain 0.3                       | Battle-tested abstractions for loaders / chains     |
| PDF parsing      | `pypdf` via `PyPDFLoader`           | Pure-Python, no system deps                         |
| Text splitting   | `RecursiveCharacterTextSplitter`    | Respects paragraph / sentence boundaries            |
| Embeddings       | `BAAI/bge-small-en` (HuggingFace)   | Free, local, CPU-friendly, strong English quality   |
| Vector store     | FAISS (CPU)                         | Fast similarity search, persistent on disk          |
| LLM              | Google Gemini 2.5 Flash             | Free tier, fast, large context window               |
| Config           | `python-dotenv`                     | `.env`-based secrets management                     |

---

## Features

- Drag-and-drop PDF upload through a Streamlit sidebar
- Automatic load → chunk → embed → index pipeline with progress feedback
- Persistent FAISS index — embeds once, reuses forever across restarts
- Chat-style Q&A with multi-turn history within a session
- Per-answer **source citations**: chunk ID, page number, and a 600-char preview
- Adjustable **Top-K** retrieval slider (1–10)
- Cached embedding model (`@st.cache_resource`) — no reload on rerun
- Modular `rag/` package — each pipeline step in its own file, fully reusable from Python scripts
- Graceful error handling for missing API key, missing index, empty PDFs

---

## Folder Structure

```
chat-with-pdf/
├── app.py                 # Streamlit UI
├── rag/                   # RAG pipeline package
│   ├── __init__.py
│   ├── loader.py          # STEP 1 — PDF → pages
│   ├── chunker.py         # STEP 2 — pages → 1000/200 chunks
│   ├── embedder.py        # STEP 3 — chunks → BGE-Small vectors
│   ├── vectorstore.py     # STEP 4 — FAISS build / save / load
│   └── retriever.py       # STEPS 5–8 — retrieve + Gemini answer
├── data/pdfs/             # Uploaded PDFs (gitignored)
├── faiss_index/           # Persisted FAISS index (gitignored)
├── requirements.txt
├── .env.example           # Template for environment variables
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The first install downloads the `BAAI/bge-small-en` model (~100 MB). One-time cost.

### 3. Add your Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```ini
GOOGLE_API_KEY=AIzaSy...your_actual_key...
```

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will open at <http://localhost:8501>.

---

## Usage

1. **Upload** a PDF using the sidebar.
2. Click **Index this PDF** — the app loads, chunks, embeds, and builds the FAISS index. This takes seconds for typical research papers, ~1 minute for a 500-page textbook.
3. **Ask** questions in the chat box. Each answer is grounded only in the document; if the answer isn't in the PDF, the model replies *"I couldn't find that in the document."*
4. Expand the **📚 Sources** section under any answer to see the exact chunks used, with chunk ID, page number, and a preview.
5. Use the **Top-K** slider to control how many chunks are retrieved per question (higher = more context, more tokens).
6. **Switch documents** by uploading a new PDF and re-indexing — the old index is replaced.
7. **🗑️ Clear index** wipes the FAISS index entirely and returns the app to a clean state.

---

## Using the Pipeline Without the UI

The `rag/` package is fully reusable as a library:

```python
from rag import (
    load_pdf, chunk_documents, get_embedder,
    build_vectorstore, build_qa_chain, answer_question,
)

pages   = load_pdf("data/pdfs/Research_Paper.pdf")
chunks  = chunk_documents(pages, chunk_size=1000, chunk_overlap=200)
store   = build_vectorstore(chunks, get_embedder())
chain   = build_qa_chain(store, top_k=4)

result = answer_question(chain, "What is the main contribution of this paper?")

print(result.answer)
for src in result.sources:
    print(f" • {src.metadata['source']} — page {src.metadata['page_number']}")
```

---

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable           | Default                  | Description                                       |
|--------------------|--------------------------|---------------------------------------------------|
| `GOOGLE_API_KEY`   | *(required)*             | Gemini API key from Google AI Studio              |
| `EMBEDDING_MODEL`  | `BAAI/bge-small-en`      | Any HuggingFace sentence-transformer model        |
| `GEMINI_MODEL`     | `gemini-2.5-flash`       | Override with `gemini-2.5-pro` for higher quality |
| `FAISS_INDEX_DIR`  | `faiss_index`            | Where the persistent FAISS index is stored        |

---

## How It Works (8 Steps)

| # | Step                  | File                  | What it does                                            |
|---|-----------------------|-----------------------|---------------------------------------------------------|
| 1 | PDF Loading           | `rag/loader.py`       | `PyPDFLoader` → one `Document` per page with metadata   |
| 2 | Chunking              | `rag/chunker.py`      | Recursive split, 1000 chars with 200-char overlap       |
| 3 | Embeddings            | `rag/embedder.py`     | `BAAI/bge-small-en` → 384-dim normalized vectors        |
| 4 | Vector DB             | `rag/vectorstore.py`  | FAISS index, persisted to `faiss_index/`                |
| 5 | Question → Embedding  | `rag/retriever.py`    | Same embedder encodes the user's question               |
| 6 | Similarity Search     | FAISS                 | Cosine similarity against all chunk vectors             |
| 7 | Retrieval             | LangChain Retriever   | Returns top-K chunks (default K=4)                      |
| 8 | LLM                   | `rag/retriever.py`    | Gemini 2.5 generates a grounded answer + citations      |

---

## Roadmap

- [x] **Day 1** — Learn embeddings, vector DBs, chunking
- [x] **Day 2** — Build PDF → FAISS → Answer pipeline
- [x] **Day 3** — Streamlit chat UI
- [x] **Day 4** — Source citations (chunk ID + page number)
- [ ] **Day 5** — Deploy on AWS EC2 (Dockerfile + nginx + systemd)
- [ ] Multi-PDF support — search across an entire library at once
- [ ] Hybrid retrieval — BM25 + dense vectors for better recall
- [ ] Streaming answers — token-by-token from Gemini for instant feedback
- [ ] Conversational memory — follow-up questions that reference prior turns

### Next project — **Resume Intelligence RAG**

This codebase evolves into a multi-document agentic system:

> Resume + Job Description → Skill-gap analysis · Interview question generator · Candidate ranking · Career recommendations

---

## Troubleshooting

| Problem                                          | Fix                                                                                              |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `RuntimeError: GOOGLE_API_KEY is not set`        | Make sure `.env` exists with a real key, then **restart Streamlit** (it loads `.env` at startup) |
| First run is very slow                           | Downloading the BGE-Small model (~100 MB) + embedding the PDF — one-time cost                    |
| FAISS load error after editing chunker settings  | Click **🗑️ Clear index** in the sidebar and re-index                                              |
| `faiss-cpu` wheel fails on Apple Silicon         | Use Python 3.10 or 3.11 in your venv — wheels aren't published for every Python version          |
| Answer says *"I couldn't find that"*             | The PDF genuinely doesn't contain the answer, **or** Top-K is too low — try raising it           |
| Streamlit shows an old PDF name after restart    | The FAISS index is cached on disk — **Clear index** to start fresh                               |

---

## Contributing

This is a learning project, but PRs and suggestions are welcome. To set up a dev environment:

```bash
git clone https://github.com/<your-username>/chat-with-pdf.git
cd chat-with-pdf
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## License

Released under the [MIT License](LICENSE).

---

## Acknowledgements

- [LangChain](https://www.langchain.com/) for the RAG abstractions
- [BAAI](https://huggingface.co/BAAI/bge-small-en) for the open-source BGE embedding model
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [Google AI Studio](https://aistudio.google.com/) for the free Gemini API
- [Streamlit](https://streamlit.io/) for the dead-simple UI framework
