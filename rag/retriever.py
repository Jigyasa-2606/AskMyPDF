"""STEPS 5–8 — Question -> Embedding -> Similarity Search -> LLM Answer.

Builds a retrieval-augmented QA chain on top of a FAISS vector store and
Google Gemini. Returns both the answer text and the source chunks so the UI
can render citations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 4

# Keep the LLM tightly grounded in the retrieved context. If it doesn't know,
# we want it to say so instead of inventing facts.
_PROMPT = PromptTemplate.from_template(
    """You are a careful research assistant. Answer the user's question
using ONLY the context below. If the answer is not in the context, say
"I couldn't find that in the document." Do not invent facts.

Context:
{context}

Question: {question}

Answer (be concise, cite specifics from the context where helpful):"""
)


@dataclass
class RagAnswer:
    answer: str
    sources: List[Document]


def _get_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "Gemini key from https://aistudio.google.com/apikey"
        )
    return ChatGoogleGenerativeAI(
        model=model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        temperature=0.2,
        google_api_key=api_key,
    )


def build_qa_chain(
    vectorstore: FAISS,
    top_k: int = DEFAULT_TOP_K,
    model_name: str | None = None,
) -> RetrievalQA:
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return RetrievalQA.from_chain_type(
        llm=_get_llm(model_name),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _PROMPT},
    )


def answer_question(chain: RetrievalQA, question: str) -> RagAnswer:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")
    result = chain.invoke({"query": question})
    return RagAnswer(
        answer=result.get("result", "").strip(),
        sources=result.get("source_documents", []) or [],
    )
