"""
test_agent.py — End-to-end smoke test for the LangGraph agent.

Poses three questions representative of the project's core use-cases:
  1. Date / validity    — titre de sejour expiry
  2. Grades             — note in a specific subject
  3. Professional CV    — experiences and skills

Run:
    python test_agent.py
"""

# load_dotenv MUST come before any project import so GROQ_API_KEY is available
# when src.agent.graph is first imported.
from dotenv import load_dotenv
load_dotenv()

import logging
import sys
import textwrap

logging.basicConfig(
    level=logging.WARNING,          # silence chromadb / sentence-transformers noise
    format="%(levelname)-8s %(name)s - %(message)s",
)

from src.retrieval.retriever import Retriever
from src.agent.graph import build_graph, run

# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

QUESTIONS = [
    {
        "id": 1,
        "label": "Titre de sejour — date d'expiration",
        "question": "Quand est-ce que le titre de sejour de Subaru expire ?",
    },
    {
        "id": 2,
        "label": "Releve de notes — note en Apprentissage par Renforcement",
        "question": "Quelle note a obtenu Subaru en Apprentissage par Renforcement ?",
    },
    {
        "id": 3,
        "label": "CV — experiences professionnelles",
        "question": "Quelles sont les experiences professionnelles de Subaru Natsuki ?",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_SEP   = "=" * 68
_DASH  = "-" * 68
_WIDTH = 68


def _safe(text: str) -> str:
    """Replace characters that Windows CP1252 console cannot display."""
    return text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8", errors="replace"
    )


def _print(text: str = "") -> None:
    print(_safe(text))


def _wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_WIDTH, initial_indent=prefix,
                         subsequent_indent=prefix)


def _print_result(case: dict, result: dict) -> None:
    _print(_SEP)
    _print(f"Q{case['id']}  {case['label']}")
    _print(_DASH)
    _print(f"Question : {case['question']}")
    _print(f"Route    : {result.get('route', '?')}  "
           f"(doc_type={result.get('doc_type')})")
    _print(f"Chunks   : {len(result.get('chunks', []))} retrieved")

    sources = result.get("sources", [])
    if sources:
        _print("Sources  :")
        for s in sources:
            _print(f"  - {s['filename']}  p.{s['page_num']}"
                   f"  [{s['doc_type']}]  score={s['score']:.3f}")

    _print()
    _print("Answer :")
    answer = result.get("answer", "(no answer)")
    for para in answer.split("\n"):
        if para.strip():
            _print(_wrap(para))
        else:
            _print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    _print(_SEP)
    _print("  Personal Knowledge Assistant — Agent Test")
    _print(_SEP)

    retriever = Retriever()
    _print(f"ChromaDB  : {retriever.count()} chunks loaded")

    agent = build_graph(retriever)
    _print("Graph     : compiled OK")
    _print()

    errors = 0
    for case in QUESTIONS:
        result = run(agent, case["question"])
        _print_result(case, result)

        answer = result.get("answer", "")
        if not answer or answer.startswith("Erreur"):
            _print(f"  [WARN] Answer missing or error for Q{case['id']}")
            errors += 1

    _print(_SEP)
    if errors:
        _print(f"  {errors}/{len(QUESTIONS)} question(s) returned an error or empty answer.")
        _print("  -> Check that GROQ_API_KEY is set in your .env file.")
        return 1

    _print(f"  All {len(QUESTIONS)}/{len(QUESTIONS)} questions answered successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
