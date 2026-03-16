# router.py
# Decides whether a question is SIMPLE or COMPLEX.
# Uses Groq (fast + free) to classify — the routing decision itself is instant.
#
# SIMPLE  → single fact, one number, direct lookup
# COMPLEX → comparison, trend, multi-step reasoning, summarisation

from langchain_core.prompts import PromptTemplate
from llm import get_groq


ROUTER_PROMPT = PromptTemplate(
    template="""Classify this financial question as SIMPLE or COMPLEX.

SIMPLE = asks for a single fact, number, or direct lookup.
Examples of SIMPLE:
- "What was revenue in 2023?"
- "What is the EPS?"
- "What was net income?"

COMPLEX = requires comparison, trend analysis, multi-step reasoning, or summarisation.
Examples of COMPLEX:
- "Compare Q1 and Q3 revenue and explain the difference"
- "What are the main financial risks and how do they affect margins?"
- "Summarise all quarterly results and identify trends"
- "Why did revenue drop and what is the outlook?"

Question: {question}

Reply with only one word — SIMPLE or COMPLEX:""",
    input_variables=["question"],
)


def classify(question: str) -> str:
    """
    Returns "SIMPLE" or "COMPLEX".
    Uses Groq so this classification takes ~0.5 seconds.
    """
    llm = get_groq()
    chain = ROUTER_PROMPT | llm
    result = chain.invoke({"question": question})
    decision = result.content.strip().upper()

    # Safety fallback — if response is unexpected, default to COMPLEX
    if "SIMPLE" in decision:
        return "SIMPLE"
    return "COMPLEX"
