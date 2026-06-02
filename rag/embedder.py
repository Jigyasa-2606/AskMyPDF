"""STEP 3 — Embeddings.

Converts text into dense vectors so we can do semantic similarity search.
Defaults to `BAAI/bge-small-en` — free, fast, runs locally on CPU, and good
quality for English. Override with the EMBEDDING_MODEL env var if you want
something else (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_MODEL = "BAAI/bge-small-en"


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embeddings client.

    Cached because loading the model weights is expensive (~100 MB download
    on first run, a few seconds to initialize on subsequent runs).
    """
    model = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    return HuggingFaceEmbeddings(
        model_name=model,
        # CPU keeps the project portable; switch to "cuda" if you have a GPU.
        model_kwargs={"device": "cpu"},
        # Normalized embeddings let FAISS use plain dot-product as cosine similarity.
        encode_kwargs={"normalize_embeddings": True},
    )
