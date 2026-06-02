"""STEP 1 — PDF Loading.

Reads a PDF from disk and returns one LangChain `Document` per page.
Each Document carries `metadata={"source": <path>, "page": <int>}` which we'll
use later for source citations in the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> List[Document]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    loader = PyPDFLoader(str(pdf_path))
    pages: List[Document] = loader.load()

    # Normalize metadata so the rest of the pipeline can rely on it.
    for page in pages:
        page.metadata.setdefault("source", str(pdf_path))
        # PyPDFLoader uses 0-indexed pages; expose a human-friendly 1-indexed number too.
        if "page" in page.metadata:
            page.metadata["page_number"] = int(page.metadata["page"]) + 1

    return pages
