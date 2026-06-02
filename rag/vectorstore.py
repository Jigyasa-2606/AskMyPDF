"""STEP 4 — Vector Database (FAISS).

Builds a FAISS index from chunks, and can persist / reload it from disk so we
don't re-embed the PDF on every app restart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_INDEX_DIR = "faiss_index"


def _index_dir(path: str | Path | None) -> Path:
    return Path(path or os.getenv("FAISS_INDEX_DIR", DEFAULT_INDEX_DIR))


def build_vectorstore(chunks: List[Document], embedder: Embeddings) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build a FAISS index from zero chunks.")
    return FAISS.from_documents(chunks, embedder)


def save_vectorstore(store: FAISS, path: str | Path | None = None) -> Path:
    out = _index_dir(path)
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    return out


def load_vectorstore(embedder: Embeddings, path: str | Path | None = None) -> FAISS | None:
    """Return the persisted FAISS index, or None if there isn't one yet."""
    src = _index_dir(path)
    if not (src / "index.faiss").exists():
        return None
    # `allow_dangerous_deserialization=True` is required by LangChain because the
    # docstore is pickled. Safe here: we only load indexes we created ourselves.
    return FAISS.load_local(
        str(src),
        embedder,
        allow_dangerous_deserialization=True,
    )
