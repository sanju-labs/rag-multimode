# FinRAG — Multimodal RAG for Financial Documents

Simple RAG system built with LangChain. Handles PDFs, images, and graph JSON files.
Currently using Gemini 2.0 Flash. One line change to switch to GPT-4o mini.

---

## Setup (do this once)

### 1. Install packages
```
pip install -r requirements.txt
```

### 2. Add your Gemini API key
Open `config.py` and replace:
```python
GOOGLE_API_KEY = "your-gemini-api-key-here"
```
Get your key free at: https://aistudio.google.com/app/apikey

---

## How to use

### Step 1 — Index your document
```
python ingest.py "Apple 2025.pdf"
```

### Step 2 — Ask questions
```
python query.py "What was Apple's revenue in 2025?"
python query.py "What are the main risks mentioned?"
python query.py "Summarise the key financial highlights."
```

---

## Switching to GPT-4o mini

Open `config.py` and change one line:
```python
PROVIDER = "openai"   # was "gemini"
```
Then add your OpenAI key in the same file. That's it — nothing else changes.

---

## File structure

```
config.py        — all settings and API keys
llm.py           — loads Gemini or OpenAI (swap here)
ingestor.py      — loads PDF / image / graph into chunks
vectorstore.py   — builds and saves the FAISS index locally
rag.py           — retrieval + generation chain
ingest.py        — run to index a file
query.py         — run to ask a question
```

## Supported file types

| File | What it does |
|------|-------------|
| .pdf | Extracts all text page by page |
| .png / .jpg | Stores as image reference for the LLM |
| .json | Reads nodes and edges as text |
