# llm.py

import config

# Embedding model cached at module level — loads ONCE when Python starts.
# Without this, it reloads from disk on every query = 8-10 second penalty.
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


def get_groq():
    """Fast free LLM — for simple factual questions."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0,
        max_tokens=200,
    )


def get_gpt():
    """Powerful LLM — for complex multi-step questions."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0,
        max_tokens=500,
    )


def get_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0,
    )
