import re
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import GROQ_MODEL, OPENAI_MODEL, TOP_K
from llm import get_gpt, get_groq
from router import classify
from vectorstore import load_vectorstore

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


SIMPLE_PROMPT = PromptTemplate(
    template="""You are a financial analyst. Answer using only the context below.
If not found, say \"Not in document.\"
Be concise and precise with numbers.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

COMPLEX_PROMPT = PromptTemplate(
    template="""You are a senior financial analyst.
Answer the question thoroughly using only the context provided below.
If comparing figures, show the numbers side by side.
If analysing trends, explain the direction and magnitude.
If summarising, cover all key points.
Cite specific figures wherever possible.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)


def _hybrid_retrieve(query: str) -> list[Document]:
    """Dense retrieval + optional BM25 rerank over retrieved chunks."""
    store = load_vectorstore()
    dense_results = store.similarity_search(query, k=TOP_K)

    if BM25_AVAILABLE and dense_results:
        tokenised = [doc.page_content.lower().split() for doc in dense_results]
        bm25 = BM25Okapi(tokenised)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, dense_results), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]

    return dense_results


def _extract_relevant_sentences(text: str, query: str, max_sentences: int) -> str:
    """Keep only the most query-relevant sentences from each chunk."""
    sentences = re.split(r"(?<=[.;\n])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return text[:500]

    query_words = set(query.lower().split())
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words)
        has_number = any(char.isdigit() for char in sentence)
        score = overlap + (2 if has_number else 0)
        scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join([s for _, s in scored[:max_sentences]])


def _build_context(chunks: list[Document], query: str, max_sentences: int) -> str:
    parts = []
    for i, doc in enumerate(chunks, 1):
        relevant = _extract_relevant_sentences(doc.page_content, query, max_sentences)
        parts.append(f"[{i}] {relevant}")
    return "\n\n".join(parts)


def _get_chunk_preview(text: str, max_chars: int = 220) -> str:
    one_line = " ".join(text.split())
    return one_line[:max_chars] + ("..." if len(one_line) > max_chars else "")


def _base_result(answer: str, chunks: list[Document], decision: str) -> dict[str, Any]:
    sources = sorted(set([doc.metadata.get("source", "unknown") for doc in chunks]))
    return {
        "answer": answer.strip(),
        "sources": sources,
        "chunks_checked": len(chunks),
        "routed_to": decision,
    }


def _run_pipeline(question: str, stream: bool = False, include_trace: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()

    t_retrieval_start = time.perf_counter()
    chunks = _hybrid_retrieve(question)
    t_retrieval = time.perf_counter() - t_retrieval_start

    t_router_start = time.perf_counter()
    decision = classify(question)
    t_router = time.perf_counter() - t_router_start

    if decision == "SIMPLE":
        llm = get_groq()
        prompt = SIMPLE_PROMPT
        max_sentences = 3
        model_name = GROQ_MODEL
    else:
        llm = get_gpt()
        prompt = COMPLEX_PROMPT
        max_sentences = 6
        model_name = OPENAI_MODEL

    t_context_start = time.perf_counter()
    context = _build_context(chunks, question, max_sentences=max_sentences)
    t_context = time.perf_counter() - t_context_start

    chain = prompt | llm

    t_generation_start = time.perf_counter()
    if stream:
        answer = ""
        for token in chain.stream({"context": context, "question": question}):
            answer += token.content
    else:
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
    t_generation = time.perf_counter() - t_generation_start

    total_time = time.perf_counter() - t0
    result = _base_result(answer=answer, chunks=chunks, decision=decision)

    if include_trace:
        result["workflow"] = {
            "question": question,
            "mode": decision,
            "model": model_name,
            "bm25_enabled": BM25_AVAILABLE,
            "steps": {
                "retrieval_seconds": round(t_retrieval, 3),
                "routing_seconds": round(t_router, 3),
                "context_seconds": round(t_context, 3),
                "generation_seconds": round(t_generation, 3),
                "total_seconds": round(total_time, 3),
            },
            "context_stats": {
                "characters": len(context),
                "estimated_tokens": max(1, len(context) // 4),
                "max_sentences_per_chunk": max_sentences,
            },
            "chunks": [
                {
                    "rank": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "preview": _get_chunk_preview(doc.page_content),
                    "length": len(doc.page_content),
                }
                for i, doc in enumerate(chunks, 1)
            ],
        }

    return result


def ask(question: str) -> dict[str, Any]:
    """Non-streaming API used by Streamlit/HTTP endpoints."""
    return _run_pipeline(question, stream=False, include_trace=False)


def ask_with_trace(question: str) -> dict[str, Any]:
    """Non-streaming API with detailed stage-by-stage workflow metadata."""
    return _run_pipeline(question, stream=False, include_trace=True)


def ask_stream(question: str) -> dict[str, Any]:
    """Streaming API used by CLI scripts."""
    return _run_pipeline(question, stream=True, include_trace=False)
