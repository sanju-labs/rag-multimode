# retriever.py
# Hybrid retrieval — combines dense vector search (FAISS) with BM25 keyword search.
#
# Why both?
#   - Dense search finds semantically similar chunks ("revenue growth")
#   - BM25 finds exact keyword matches ("$94.9B", "Q3 2024", "net sales")
#   - Financial documents need both — numbers and tickers are missed by vectors alone
#
# After retrieving candidates from both, we pick the best answer by
# asking the LLM to score each chunk's relevance to the question.

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from vectorstore import load_vectorstore
from config import TOP_K


class HybridRetriever:
    def __init__(self, chunks: list[Document]):
        """
        Takes the full list of chunks.
        Builds both a FAISS index and a BM25 index over them.
        """
        self.chunks = chunks

        # Dense retriever (semantic)
        self.vectorstore = load_vectorstore()
        self.dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )

        # BM25 retriever (keyword)
        tokenised = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenised)

    def retrieve(self, query: str) -> list[Document]:
        """
        Runs both retrievers and merges results.
        Deduplicates by content so the LLM doesn't see the same chunk twice.
        """
        # Dense results
        dense_results = self.dense_retriever.invoke(query)

        # BM25 results
        bm25_results = self._bm25_search(query, TOP_K)

        # Merge and deduplicate
        seen = set()
        merged = []
        for doc in dense_results + bm25_results:
            key = doc.page_content[:100]   # first 100 chars as dedup key
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # Get top k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_indices]
