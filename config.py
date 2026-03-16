# config.py
# All settings in one place. Only touch this file to change models or keys.
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("PINECONE_API_KEY")

# --- API Keys ---
GOOGLE_API_KEY = "your-gemini-api-key-here"
OPENAI_API_KEY = "GPT API"
GROQ_API_KEY    = "Groq API"

# --- Models ---
GEMINI_MODEL = "gemini-1.5-flash-latest"
OPENAI_MODEL = "gpt-4o-mini"     #used for complex questions
GROQ_MODEL      = "llama-3.1-8b-instant"      # used for simple questions

# --- RAG settings ---
CHUNK_SIZE = 1000      # characters per chunk — good for dense financial text
CHUNK_OVERLAP = 150    # overlap so figures spanning paragraphs aren't lost
TOP_K = 5              # chunks retrieved — kept at 6 for full accuracy
