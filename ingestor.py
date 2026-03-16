# ingestor.py
# Loads and splits documents into chunks.
# Supports: PDF, images (PNG/JPG), and graph JSON files.

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(file_path):
    """Detects file type and loads it into LangChain Document format."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix in [".png", ".jpg", ".jpeg"]:
        return _load_image(file_path)
    elif suffix == ".json":
        return _load_graph(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF, image, or JSON.")


def split_documents(docs):
    """Splits documents into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# --- private loaders ---

def _load_pdf(file_path):
    """Loads each page of a PDF as a separate document."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def _load_image(file_path):
    """
    Stores the image path as a document.
    The LLM will see it as a chart/diagram reference in the context.
    For a future upgrade: replace this with a vision model call.
    """
    return [Document(
        page_content=f"[Image: {Path(file_path).name}] This is a chart or diagram from the financial report.",
        metadata={"source": file_path, "type": "image"},
    )]


def _load_graph(file_path):
    """
    Loads a JSON graph file with nodes and edges.
    Expected format: {"nodes": [...], "edges": [...]}
    Converts the graph into readable text lines for the LLM.
    """
    data = json.loads(Path(file_path).read_text())
    lines = []

    for node in data.get("nodes", []):
        lines.append(f"{node.get('label', node['id'])} (type: {node.get('type', 'entity')})")

    for edge in data.get("edges", []):
        lines.append(f"{edge['source']} -> {edge['target']} [{edge.get('relation', 'related')}]")

    return [Document(
        page_content="\n".join(lines),
        metadata={"source": file_path, "type": "graph"},
    )]
