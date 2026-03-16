# config.py
# All settings in one place. Only touch this file to change models or keys.
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Models ---
GEMINI_MODEL = "gemini-1.5-flash-latest"
OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL = "llama-3.1-8b-instant"

# --- RAG settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5