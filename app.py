"""Chat-with-PDF — Streamlit front-end.

Run with:
    streamlit run app.py

Flow:
    1. Upload a PDF (or reuse one already in data/pdfs/).
    2. We load -> chunk -> embed -> build a FAISS index (cached on disk).
    3. Ask questions; we retrieve the top-k chunks and ask Gemini to answer.
    4. Each answer shows the source chunks + page numbers used.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import (
    answer_question,
    build_qa_chain,
    build_vectorstore,
    chunk_documents,
    get_embedder,
    load_pdf,
    load_vectorstore,
    save_vectorstore,
)

load_dotenv()

PDF_DIR = Path("data/pdfs")
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "faiss_index"))
PDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDF")
st.caption("Vanilla RAG · PyPDFLoader · BGE-Small · FAISS · Gemini 2.5")


# ---------- cached pipeline pieces ----------

@st.cache_resource(show_spinner="Loading embedding model…")
def _embedder():
    return get_embedder()


def _index_meta_path() -> Path:
    return INDEX_DIR / "active_pdf.txt"


def _active_pdf_name() -> str | None:
    p = _index_meta_path()
    return p.read_text().strip() if p.exists() else None


def _set_active_pdf(name: str) -> None:
    _index_meta_path().write_text(name)


def _reset_index() -> None:
    if INDEX_DIR.exists():
        for child in INDEX_DIR.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def ingest_pdf(pdf_path: Path) -> int:
    """Run the loader -> chunker -> embedder -> FAISS pipeline. Returns chunk count."""
    embedder = _embedder()

    with st.status("Ingesting PDF…", expanded=True) as status:
        st.write("📥 Loading PDF…")
        pages = load_pdf(pdf_path)
        st.write(f"   • {len(pages)} page(s) loaded")

        st.write("✂️  Chunking text (1000 / 200 overlap)…")
        chunks = chunk_documents(pages)
        st.write(f"   • {len(chunks)} chunks produced")

        st.write("🧠 Generating embeddings + building FAISS index…")
        store = build_vectorstore(chunks, embedder)

        st.write("💾 Saving index to disk…")
        _reset_index()
        save_vectorstore(store, INDEX_DIR)
        _set_active_pdf(pdf_path.name)

        status.update(label=f"Indexed {pdf_path.name} ({len(chunks)} chunks)", state="complete")

    return len(chunks)


def get_qa_chain():
    embedder = _embedder()
    store = load_vectorstore(embedder, INDEX_DIR)
    if store is None:
        return None
    top_k = st.session_state.get("top_k", 4)
    return build_qa_chain(store, top_k=top_k)


# ---------- sidebar: upload / settings ----------

with st.sidebar:
    st.header("⚙️ Setup")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "`GOOGLE_API_KEY` is not set.\n\n"
            "Copy `.env.example` to `.env` and add your Gemini key "
            "from [Google AI Studio](https://aistudio.google.com/apikey)."
        )

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None:
        target = PDF_DIR / uploaded.name
        target.write_bytes(uploaded.getvalue())
        st.success(f"Saved → {target}")
        if st.button("Index this PDF", type="primary", use_container_width=True):
            ingest_pdf(target)
            st.rerun()

    st.divider()
    st.subheader("Retrieval")
    st.slider("Top-K chunks", 1, 10, 4, key="top_k")

    st.divider()
    active = _active_pdf_name()
    if active:
        st.success(f"Active index: **{active}**")
    else:
        st.info("No index yet. Upload a PDF and click *Index this PDF*.")

    if active and st.button("🗑️ Clear index"):
        _reset_index()
        st.rerun()


# ---------- main: ask questions ----------

active = _active_pdf_name()
if not active:
    st.info("👈 Upload and index a PDF to get started.")
    st.stop()

st.subheader(f"Ask anything about **{active}**")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["sources"]:
            with st.expander(f"📚 Sources ({len(turn['sources'])})"):
                for src in turn["sources"]:
                    meta = src.metadata
                    page = meta.get("page_number", meta.get("page", "?"))
                    chunk_id = meta.get("chunk_id", "?")
                    source_path = meta.get("source", "uploaded.pdf")
                    st.markdown(
                        f"**Chunk #{chunk_id}** — _{Path(source_path).name}_, page {page}"
                    )
                    preview = src.page_content.strip().replace("\n", " ")
                    if len(preview) > 600:
                        preview = preview[:600] + "…"
                    st.write(preview)
                    st.divider()

question = st.chat_input("e.g. What is the main contribution of this paper?")
if question:
    chain = get_qa_chain()
    if chain is None:
        st.error("No FAISS index found. Re-index your PDF from the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and asking Gemini…"):
            try:
                result = answer_question(chain, question)
            except Exception as exc:  # surfaces missing API key, network errors, etc.
                st.error(f"Something went wrong: {exc}")
                st.stop()

        st.markdown(result.answer)
        if result.sources:
            with st.expander(f"📚 Sources ({len(result.sources)})"):
                for src in result.sources:
                    meta = src.metadata
                    page = meta.get("page_number", meta.get("page", "?"))
                    chunk_id = meta.get("chunk_id", "?")
                    source_path = meta.get("source", "uploaded.pdf")
                    st.markdown(
                        f"**Chunk #{chunk_id}** — _{Path(source_path).name}_, page {page}"
                    )
                    preview = src.page_content.strip().replace("\n", " ")
                    if len(preview) > 600:
                        preview = preview[:600] + "…"
                    st.write(preview)
                    st.divider()

    st.session_state.history.append(
        {"question": question, "answer": result.answer, "sources": result.sources}
    )
