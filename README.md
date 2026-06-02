# Chat with PDF — Vanilla RAG

Upload a PDF, ask a question, get an answer grounded in the document — with
the exact source chunks shown next to the answer.

```
PDF → PyPDFLoader → Chunker → BGE-Small Embeddings → FAISS
                                                       │
                                  Question → Embedding ┘
                                       │
                                  Top-K chunks → Gemini 2.5 → Answer + Citations
```

This is **vanilla RAG** — one retrieval, one LLM call. (Agentic RAG, with
multi-step tool use, comes later.)

## Tech stack

| Layer        | Choice                        |
|--------------|-------------------------------|
| Frontend     | Streamlit                     |
| Backend      | Python 3.10+                  |
| RAG glue     | LangChain                     |
| Embeddings   | `BAAI/bge-small-en` (local)   |
| Vector DB    | FAISS (CPU, on-disk)          |
| LLM          | Gemini 2.5 Flash              |

## Folder structure

```
chat-with-pdf/
├── app.py                 # Streamlit UI
├── rag/
│   ├── loader.py          # STEP 1 — PDF → pages
│   ├── chunker.py         # STEP 2 — pages → chunks
│   ├── embedder.py        # STEP 3 — chunks → vectors
│   ├── vectorstore.py     # STEP 4 — FAISS build/save/load
│   └── retriever.py       # STEPS 5–8 — retrieve + Gemini answer
├── data/pdfs/             # uploaded PDFs (gitignored)
├── faiss_index/           # persisted FAISS index (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Create a virtual env and install deps**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   First install pulls down the BGE-Small model (~100 MB). One-time only.

2. **Add your Gemini key**

   Get a free key from <https://aistudio.google.com/apikey>, then:

   ```bash
   cp .env.example .env
   # edit .env, paste your key into GOOGLE_API_KEY
   ```

3. **Run it**

   ```bash
   streamlit run app.py
   ```

   Streamlit opens at <http://localhost:8501>.

## How to use it

1. Drag a PDF into the sidebar uploader.
2. Click **Index this PDF**. The app loads → chunks → embeds → builds FAISS.
   This is a one-time cost per PDF; the index is cached in `faiss_index/`.
3. Ask questions in the chat box. Each answer shows the source chunks used,
   with page numbers, so you can verify nothing was hallucinated.
4. Use the sidebar slider to change **Top-K** (how many chunks to feed the
   LLM). Higher = more context, slower, more tokens.

## Using the pipeline from Python (no UI)

```python
from rag import load_pdf, chunk_documents, get_embedder, build_vectorstore, build_qa_chain, answer_question

pages = load_pdf("data/pdfs/Research_Paper.pdf")
chunks = chunk_documents(pages)
store = build_vectorstore(chunks, get_embedder())
chain = build_qa_chain(store, top_k=4)

result = answer_question(chain, "What is the main contribution?")
print(result.answer)
for src in result.sources:
    print(" -", src.metadata.get("source"), "p.", src.metadata.get("page_number"))
```

## Roadmap

- [x] **Day 1** — Learn embeddings, vector DBs, chunking
- [x] **Day 2** — Build PDF → FAISS → Answer pipeline
- [x] **Day 3** — Streamlit UI
- [x] **Day 4** — Source citations
- [ ] **Day 5** — Deploy on AWS EC2

After Day 5, this evolves into **Resume Intelligence RAG**:
resume + JD → skill gap analysis → interview questions → candidate ranking.

## Troubleshooting

- **`GOOGLE_API_KEY is not set`** — Make sure `.env` is in the project root
  and contains a valid key. Restart Streamlit after editing it.
- **First run is slow** — Downloading the BGE-Small model and embedding the
  PDF. Subsequent runs reuse the cached model and the saved FAISS index.
- **FAISS load error after editing chunks** — Click *Clear index* in the
  sidebar, then re-index the PDF.
- **`faiss-cpu` install fails on Apple Silicon** — Use Python 3.10 or 3.11
  in your venv; faiss wheels are not yet published for every Python version.
