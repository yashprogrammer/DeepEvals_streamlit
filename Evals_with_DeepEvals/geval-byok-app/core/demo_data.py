"""core/demo_data.py — bundled sample data for the RAG / Agent / Multiturn demo sections.

Ported verbatim from the course notebooks so students see the same corpus/tools/examples in
the notebook and the live app: 01_rag_evals_with_deepeval.ipynb,
02_agent_evals_with_deepeval.ipynb, 03_chatbot_conversation_evals_with_deepeval.ipynb.
"""
from __future__ import annotations

import random

# ──────────────────────────────────────────────────────────────────────────────
# RAG demo data (01_rag_evals_with_deepeval.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
RAG_CORPUS = [
    {"title": "Tokens", "content": (
        "Large language models split text into tokens, common character sequences roughly 4 "
        "characters or three-quarters of a word long. Both the prompt and the output are counted "
        "in tokens, and pricing and context limits are measured in tokens.")},
    {"title": "Embeddings", "content": (
        "An embedding is a fixed-length vector that represents the meaning of a piece of text. "
        "Texts with similar meaning have vectors that are close together, usually measured by "
        "cosine similarity, which is what lets a system do semantic search.")},
    {"title": "Retrieval-Augmented Generation", "content": (
        "RAG grounds an LLM's answer in external documents. At query time the system retrieves "
        "the most relevant chunks and passes them to the model as context, which reduces "
        "hallucination and lets you update knowledge without retraining.")},
    {"title": "Hallucination", "content": (
        "A hallucination is fluent, confident text that is factually wrong or unsupported by its "
        "sources. It happens because the model predicts likely text rather than looking facts up. "
        "Grounding answers in retrieved context is the main defense.")},
    {"title": "AI agents", "content": (
        "An AI agent is an LLM given a goal, tools, and a loop: plan, call a tool, observe the "
        "result, decide the next step, until the task is done.")},
]


def retrieve(query: str, k: int = 2) -> list[str]:
    """Naive retriever: rank docs by keyword overlap with the query. Good enough to teach evals."""
    q_words = set(query.lower().split())
    scored = []
    for doc in RAG_CORPUS:
        doc_words = set((doc["title"] + " " + doc["content"]).lower().split())
        overlap = len(q_words & doc_words)
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc["content"] for _, doc in scored[:k]]


RAG_SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. If the context doesn't contain the answer, say you "
    "don't have that information. Be concise (1-2 sentences)."
)

RAG_EXAMPLE_QUESTIONS = [
    {"input": "What is a token in an LLM?",
     "expected_output": "A token is a common character sequence, roughly 4 characters, used to "
                         "measure both prompt and output length."},
    {"input": "How does RAG reduce hallucination?",
     "expected_output": "RAG retrieves relevant chunks and passes them as context, grounding the "
                         "answer in real documents instead of only the model's training."},
    {"input": "How do I containerize a model for deployment with Docker?",
     "expected_output": ""},  # deliberately outside the tiny knowledge base
]


# ──────────────────────────────────────────────────────────────────────────────
# Agent demo data (02_agent_evals_with_deepeval.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
def search_docs(query: str = "") -> str:
    """A real (tiny) RAG pipeline, not a canned string: reuses the same keyword retriever and
    RAG_CORPUS as the RAG Evals section, so the agent's tool call actually retrieves and the
    trace in the Interact tab reflects a genuine retrieval, not a scripted response.
    """
    if not query.strip():
        return "No query given -- nothing retrieved."
    passages = retrieve(query, k=2)
    if not passages:
        return f"No documentation found for '{query}'."
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(passages, 1))
    return f"Docs result for '{query}':\n{numbered}"


def web_search(query: str = "") -> str:
    """Real, live DuckDuckGo web search (no API key needed -- ddgs talks to DDG directly), so the
    agent's tool call actually executes instead of returning canned text. Swapped in for a
    Python-sandbox tool, which isn't safe to run inside a shared Streamlit Community Cloud process.
    """
    if not query.strip():
        return "No query given -- nothing searched."
    from ddgs import DDGS

    try:
        results = DDGS().text(query, max_results=3)
    except Exception as e:
        return f"Web search failed: {type(e).__name__}: {e}"
    if not results:
        return f"No web results found for '{query}'."
    numbered = "\n".join(
        f"{i}. {r.get('title', '')} -- {r.get('body', '')} ({r.get('href', '')})"
        for i, r in enumerate(results, 1)
    )
    return f"Web search results for '{query}':\n{numbered}"


def escalate_to_human(reason: str = "") -> str:
    return f"Escalated to a human engineer. Ticket AI-{random.randint(1000, 9999)}. Reason: {reason}"


AGENT_TOOLS = {"search_docs": search_docs, "web_search": web_search, "escalate_to_human": escalate_to_human}

AGENT_TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search_docs",
        "description": "Search internal documentation for a concept or policy.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_search",
        "description": "Search the public web (DuckDuckGo) for up-to-date information and return the top results.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "escalate_to_human",
        "description": "Escalate the issue to a human engineer.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]

AGENT_SYSTEM_PROMPT = (
    "You are a GenAI developer assistant. Before acting, briefly state your plan in one short "
    "sentence starting with 'Plan:', then call the tool(s) that actually accomplish the task -- "
    "do not guess something you can look up or search for."
)

AGENT_GOAL = (
    "Help developers with GenAI questions by looking things up in the docs, searching the web "
    "for anything the docs don't cover, and escalating to a human when neither resolves the issue."
)

# Ground-truth scenarios for the Agent Evals demo. Each is a sequence of user turns (most are
# single-turn; one is multi-turn) plus the tools the agent should call and the goal it should
# reach across the whole scenario.
AGENT_SCENARIOS = [
    {
        "name": "Fine-tuning vs prompting",
        "turns": ["What do the docs say about the difference between fine-tuning and prompting?"],
        "expected_tools": ["search_docs"],
        "goal": "Explain the difference between fine-tuning and prompting using the documentation.",
    },
    {
        "name": "Search the web",
        "turns": ["What's the latest stable version of the DeepEval Python package? "
                  "Search the web and tell me."],
        "expected_tools": ["web_search"],
        "goal": "Search the web and report the latest DeepEval version.",
    },
    {
        "name": "Docs first, then escalate",
        "turns": ["My fine-tuning job keeps failing silently. Check the docs for known issues, "
                  "and if that doesn't explain it, get a human to look at it."],
        "expected_tools": ["search_docs", "escalate_to_human"],
        "goal": "Check the documentation first, then escalate to a human only if it doesn't "
                "resolve the issue.",
    },
    {
        "name": "Debug, then escalate (multi-turn)",
        "turns": [
            "Can you check if there's a known issue with fine-tuning jobs failing silently?",
            "That didn't help. Can you search the web for known Groq fine-tuning silent-failure "
            "issues?",
            "Still stuck -- please escalate this to a human.",
        ],
        "expected_tools": ["search_docs", "web_search", "escalate_to_human"],
        "goal": "Investigate via the docs and a web search before escalating to a human.",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Multiturn demo data (03_chatbot_conversation_evals_with_deepeval.ipynb,
# assistant system prompt from 05_synthetic_conversation_data_generation.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONVERSATION: list[tuple[str, str]] = [
    ("user", "Hey, what's a context window in an LLM?"),
    ("assistant", "The context window is the maximum number of tokens the model can consider at "
                  "once -- covering the system prompt, the conversation history, and its own "
                  "reply."),
    ("user", "Does a bigger context window always mean better answers?"),
    ("assistant", "Not necessarily. A bigger window gives more room for context, but the model "
                  "still has to use it well -- very long contexts can dilute attention, so "
                  "retrieval quality matters as much as sheer size."),
    ("user", "Cool, unrelated, but do you know a good pizza place nearby?"),
    ("assistant", "I'm a GenAI concepts assistant, so I can't help with restaurant "
                  "recommendations -- happy to keep answering questions about LLMs and RAG "
                  "though!"),
]

DEFAULT_CHATBOT_ROLE = (
    "a GenAI concepts assistant that explains LLM and RAG topics, and declines unrelated requests"
)
DEFAULT_SCENARIO = "A developer is casually asking a documentation chatbot about context windows."
DEFAULT_EXPECTED_OUTCOME = "The user understands what a context window is and its trade-offs."
MULTITURN_RELEVANT_TOPICS = ["large language models", "RAG", "AI agents", "GenAI concepts"]
MULTITURN_ASSISTANT_SYSTEM = "You are a helpful GenAI concepts assistant. Be concise (1-2 sentences)."
