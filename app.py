# load_dotenv before any project import so GROQ_API_KEY is available
# when src modules are first imported.
from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime, timezone

import streamlit as st

# ── Page config — must be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils.metadata import load as load_metadata
from src.retrieval.retriever import Retriever
from src.agent.graph import build_graph, run as agent_run

# ──────────────────────────────────────────────────────────────────────────────
# Doc-type display config
# ──────────────────────────────────────────────────────────────────────────────

_ICON = {
    "cv":           "👤",
    "transcript":   "📊",
    "titre_sejour": "🪪",
    "contract":     "📝",
    "planning":     "📅",
    "attestation":  "📋",
    "unknown":      "📄",
}

_COLOR = {
    "cv":           "#1a56db",
    "transcript":   "#057a55",
    "titre_sejour": "#c27803",
    "contract":     "#7e3af2",
    "planning":     "#0694a2",
    "attestation":  "#e02424",
    "unknown":      "#6b7280",
}


def _icon(doc_type: str) -> str:
    return _ICON.get(doc_type, "📄")


def _color(doc_type: str) -> str:
    return _COLOR.get(doc_type, "#6b7280")


# ──────────────────────────────────────────────────────────────────────────────
# Cached heavy resources  (loaded once per Streamlit server process)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modele et de la base vectorielle...")
def get_agent():
    """Load the Retriever and compile the LangGraph agent once."""
    retriever = Retriever()
    graph = build_graph(retriever)
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso).astimezone(timezone.utc)
        return dt.strftime("%d/%m/%Y  %H:%M UTC")
    except ValueError:
        return iso


def render_sidebar() -> None:
    meta = load_metadata()

    with st.sidebar:
        st.title("📚 Knowledge Base")

        # ── Status ──────────────────────────────────────────────────────────
        if meta["num_chunks"] > 0:
            st.success("Base vectorielle chargee", icon="✅")
        else:
            st.warning(
                "Base vide — lance `python ingest.py` pour indexer des documents.",
                icon="⚠️",
            )

        # ── API key ──────────────────────────────────────────────────────────
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY manquante — copie `.env.example` en `.env`.", icon="🔑")

        st.divider()

        # ── Stats ────────────────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        c1.metric("Documents", meta["num_docs"])
        c2.metric("Chunks", meta["num_chunks"])
        st.caption(f"Derniere MAJ : {_fmt_date(meta.get('last_updated'))}")

        st.divider()

        # ── File list ────────────────────────────────────────────────────────
        files = meta.get("indexed_files", [])
        if files:
            st.subheader("Fichiers indexes")
            for f in files:
                icon  = _icon(f["doc_type"])
                color = _color(f["doc_type"])
                st.markdown(
                    f"""<div style="
                        border-left:3px solid {color};
                        padding:6px 10px;
                        border-radius:3px;
                        margin-bottom:6px;
                        background:#f8f9fa;
                    ">
                    <span style="font-weight:600">{icon}&nbsp;{f['filename']}</span><br>
                    <span style="font-size:0.78rem;color:#555">
                        {f['doc_type']}&nbsp;&middot;&nbsp;{f['num_chunks']} chunks
                    </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Aucun fichier indexe.")

        st.divider()

        # ── Clear chat ───────────────────────────────────────────────────────
        if st.button("🗑️  Effacer le chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption("Powered by Groq · LangGraph · ChromaDB")


# ──────────────────────────────────────────────────────────────────────────────
# Source cards
# ──────────────────────────────────────────────────────────────────────────────

def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    label = f"📎 {len(sources)} source{'s' if len(sources) > 1 else ''} utilisee{'s' if len(sources) > 1 else ''}"
    with st.expander(label, expanded=True):
        n_cols = min(len(sources), 3)
        cols = st.columns(n_cols)
        for i, src in enumerate(sources):
            icon  = _icon(src["doc_type"])
            color = _color(src["doc_type"])
            score_pct = int(src["score"] * 100)
            with cols[i % n_cols]:
                st.markdown(
                    f"""<div style="
                        border-left:4px solid {color};
                        padding:8px 12px;
                        border-radius:4px;
                        background:#f8fafc;
                        margin-bottom:8px;
                    ">
                    <div style="font-weight:600;font-size:0.9rem">
                        {icon}&nbsp;{src['filename']}
                    </div>
                    <div style="font-size:0.78rem;color:#555;margin-top:3px">
                        Page&nbsp;{src['page_num']}
                        &nbsp;&middot;&nbsp;{src['doc_type']}
                        &nbsp;&middot;&nbsp;pertinence&nbsp;{score_pct}%
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────────────────────
# Chat history helpers
# ──────────────────────────────────────────────────────────────────────────────

def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    st.title("💬 Personal Knowledge Assistant")
    st.caption(
        "Posez vos questions sur les documents indexes. "
        "L'agent choisit automatiquement le bon outil et cite ses sources."
    )

    _init_session()
    _render_history()

    if prompt := st.chat_input("Votre question..."):

        # ── Display user message ─────────────────────────────────────────────
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # ── Run agent ────────────────────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.spinner("Recherche et generation en cours..."):
                try:
                    graph  = get_agent()
                    result = agent_run(graph, prompt)
                    answer  = result.get("answer") or ""
                    sources = result.get("sources") or []
                    route   = result.get("route", "")
                    dtype   = result.get("doc_type")
                except Exception as exc:
                    answer  = f"Erreur inattendue : {exc}"
                    sources = []
                    route   = ""
                    dtype   = None

            # Answer
            st.markdown(answer if answer else "_Aucune reponse generee._")

            # Routing badge (subtle, for transparency)
            if route:
                badge_label = f"tool : {route}"
                if dtype:
                    badge_label += f"  ·  filtre : {dtype}"
                st.caption(f"🔀 {badge_label}")

            # Sources
            render_sources(sources)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
        })


main()
