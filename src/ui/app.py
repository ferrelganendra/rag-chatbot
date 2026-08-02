"""Streamlit UI for RAG QA Engine."""

import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from retrieval.qa_engine import QAEngine
from retrieval.searcher import Searcher


st.set_page_config(
    page_title="RAG QA Engine",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG QA Engine")
st.caption("Ask questions over technical documents — powered by local LLM + vector search")


@st.cache_resource
def load_engine():
    return QAEngine(searcher=Searcher())


engine = load_engine()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Retrieved chunks", 1, 10, 5)
    st.header("📄 Indexed Documents")
    try:
        col = engine.searcher.collection
        sources = col.get()["metadatas"]
        unique_sources = sorted({m["source"] for m in sources if m})
        for s in unique_sources:
            st.text(f"• {s}")
    except Exception:
        st.text("No documents indexed")
    st.divider()
    st.caption("Qwen3 0.6B + all-MiniLM-L6-v2 + ChromaDB")

# Main
question = st.text_input("Ask a question about the documents:", placeholder="e.g., What is RAG?")

if question:
    with st.spinner("Retrieving context..."):
        result = engine.answer(question)

    # Answer
    st.header("Answer")
    st.write(result["answer"])

    # Sources
    st.divider()
    st.subheader("📖 Sources")

    cols = st.columns(min(len(result["sources"]), 3))
    for i, src in enumerate(result["sources"]):
        with cols[i % 3]:
            with st.container(border=True):
                st.caption(f"**{src['document']}**")
                st.metric("Relevance", f"{src['score']:.3f}")
                with st.expander("View chunk"):
                    st.text(result["context_chunks"][i])
