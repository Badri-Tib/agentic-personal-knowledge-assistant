"""
graph.py — LangGraph agent graph for the Personal Knowledge Assistant.

Graph topology:

    START
      └─→ [router_node]  — heuristic routing, sets route + doc_type
              ├─(search_kb)────→ [search_node]   — semantic search, any doc_type
              └─(extract_dates)→ [dates_node]    — scoped to titre_sejour/contract
                                       ↓
                              [generate_node]    — Groq API call, answer + sources
                                       ↓
                                      END

Usage::

    from src.retrieval.retriever import Retriever
    from src.agent.graph import build_graph, run

    retriever = Retriever()
    agent = build_graph(retriever)

    result = run(agent, "Quand est-ce que le titre de sejour expire ?")
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.tools import (
    ROUTE_EXTRACT_DATES,
    ROUTE_SEARCH_KB,
    decide_route,
    extract_dates,
    format_context,
    search_kb,
)
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Groq config
# ──────────────────────────────────────────────────────────────────────────────

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_SYSTEM_PROMPT = """\
Tu es un assistant de connaissance personnelle. \
Reponds aux questions en te basant UNIQUEMENT sur les extraits de documents fournis.
Regles :
- Cite toujours tes sources (nom du fichier, numero de page).
- Si la reponse n'est pas dans les documents, dis-le clairement sans inventer.
- Sois concis et precis.
- Reponds dans la meme langue que la question posee.\
"""

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    question: str                       # user query (required at invoke time)
    route: str                          # ROUTE_SEARCH_KB | ROUTE_EXTRACT_DATES
    doc_type: str | list[str] | None    # metadata filter decided by router
    chunks: list[dict[str, Any]]        # retrieved chunks (set by search/dates node)
    answer: str                         # final LLM answer
    sources: list[dict[str, Any]]       # deduplicated source list for the UI


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """Decide which retrieval path to take, without an LLM call."""
    route, doc_type = decide_route(state["question"])
    logger.info(
        "router: %r → route=%s  doc_type=%s",
        state["question"][:70],
        route,
        doc_type,
    )
    return {"route": route, "doc_type": doc_type}


def _make_search_node(retriever: Retriever):
    def search_node(state: AgentState) -> AgentState:
        """Semantic search with optional doc_type filter."""
        chunks = search_kb(
            retriever,
            query=state["question"],
            k=5,
            doc_type=state.get("doc_type"),
        )
        logger.info("search_node: %d chunk(s) retrieved", len(chunks))
        return {"chunks": chunks}
    return search_node


def _make_dates_node(retriever: Retriever):
    def dates_node(state: AgentState) -> AgentState:
        """Date-targeted retrieval on titre_sejour / contract docs."""
        chunks = extract_dates(
            retriever,
            query=state["question"],
            k=5,
        )
        logger.info("dates_node: %d chunk(s) retrieved", len(chunks))
        return {"chunks": chunks}
    return dates_node


def generate_node(state: AgentState) -> AgentState:
    """Format context and call Groq to produce the final answer."""
    try:
        from groq import Groq
    except ImportError:
        return {
            "answer": "Erreur : la librairie `groq` n'est pas installee (pip install groq).",
            "sources": [],
        }

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "answer": (
                "Erreur : GROQ_API_KEY non definie. "
                "Copie .env.example en .env et renseigne ta cle API Groq."
            ),
            "sources": [],
        }

    chunks: list[dict[str, Any]] = state.get("chunks") or []
    context = format_context(chunks)

    user_prompt = f"Question : {state['question']}\n\nDocuments :\n{context}"

    logger.info("generate_node: calling Groq (%s) with %d chunk(s)", GROQ_MODEL, len(chunks))

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        answer = f"Erreur lors de l'appel a l'API Groq : {exc}"

    sources = _deduplicate_sources(chunks)
    logger.info("generate_node: answer %d chars, %d source(s)", len(answer), len(sources))
    return {"answer": answer, "sources": sources}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _route_edge(state: AgentState) -> str:
    """Conditional edge: read the route chosen by router_node."""
    return state["route"]


def _deduplicate_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one entry per (filename, page_num) pair, keeping the best score."""
    seen: dict[tuple, dict] = {}
    for c in chunks:
        key = (c["filename"], c["page_num"])
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = {
                "filename":    c["filename"],
                "source_file": c["source_file"],
                "page_num":    c["page_num"],
                "doc_type":    c["doc_type"],
                "score":       c["score"],
            }
    return sorted(seen.values(), key=lambda s: -s["score"])


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(retriever: Retriever):
    """
    Compile the LangGraph agent and return it.

    The retriever is injected into search/dates nodes via closures — the
    compiled graph itself is stateless and can be created once at startup.

    Args:
        retriever: A ready Retriever (model + ChromaDB connected).

    Returns:
        Compiled LangGraph graph. Invoke with:
        ``graph.invoke({"question": "..."})``
    """
    graph = StateGraph(AgentState)

    graph.add_node("router",   router_node)
    graph.add_node("search",   _make_search_node(retriever))
    graph.add_node("dates",    _make_dates_node(retriever))
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_edge,
        {
            ROUTE_SEARCH_KB:     "search",
            ROUTE_EXTRACT_DATES: "dates",
        },
    )
    graph.add_edge("search",   "generate")
    graph.add_edge("dates",    "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def run(graph, question: str) -> dict[str, Any]:
    """
    Invoke the compiled graph with a question and return the full state.

    Args:
        graph: Output of build_graph().
        question: Natural-language question string.

    Returns:
        Dict with keys: question, route, doc_type, chunks, answer, sources.
    """
    initial_state: AgentState = {
        "question": question,
        "route":    "",
        "doc_type": None,
        "chunks":   [],
        "answer":   "",
        "sources":  [],
    }
    return graph.invoke(initial_state)
