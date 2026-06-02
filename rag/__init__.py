"""RAG pipeline: PDF -> chunks -> embeddings -> FAISS -> retriever -> LLM."""

from .loader import load_pdf
from .chunker import chunk_documents
from .embedder import get_embedder
from .vectorstore import build_vectorstore, load_vectorstore, save_vectorstore
from .retriever import build_qa_chain, answer_question

__all__ = [
    "load_pdf",
    "chunk_documents",
    "get_embedder",
    "build_vectorstore",
    "load_vectorstore",
    "save_vectorstore",
    "build_qa_chain",
    "answer_question",
]
