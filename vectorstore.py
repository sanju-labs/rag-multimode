# vectorstore.py

import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

STORE_PATH = "faiss_index"

# FAISS store cached at module level — loads index once, not on every query
_store = None


def build_vectorstore(chunks):
    """Creates a FAISS index from chunks and saves it locally."""
    from llm import get_embeddings
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(STORE_PATH)
    print(f"Index saved — {len(chunks)} chunks stored in '{STORE_PATH}/'")
    return store


def load_vectorstore():
    """Loads the FAISS index — cached after first load."""
    global _store
    if _store is not None:
        return _store

    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError(
            f"No index found. Run: python ingest.py \"your_file.pdf\""
        )

    from llm import get_embeddings
    embeddings = get_embeddings()
    _store = FAISS.load_local(
        STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return _store
