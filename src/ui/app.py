"""Streamlit UI — production-style RAG QA interface with streaming, chat history, source cards."""

import sys
from pathlib import Path
import streamlit as st

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from retrieval.qa_engine import QAEngine
from retrieval.searcher import Searcher

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DocQ — RAG QA Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS polish ───────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%); }
    section[data-testid="stSidebar"] { background: rgba(15,17,23,0.95); border-right: 1px solid rgba(255,255,255,0.06); }
    .stChatMessage { background: transparent !important; }
    [data-testid="stChatMessage"] { border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: 0.5rem; }
    [data-testid="stChatMessage"]:has(.chat-user) { background: linear-gradient(135deg, #2563eb15, #7c3aed15); border: 1px solid rgba(99,102,241,0.2); }
    [data-testid="stChatMessage"]:has(.chat-assistant) { background: linear-gradient(135deg, #1e293b, #1a1d2e); border: 1px solid rgba(255,255,255,0.06); }
    .source-card { background: linear-gradient(135deg, #1e293b, #111827); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .source-card-high { border-left: 3px solid #22c55e; }
    .source-card-mid  { border-left: 3px solid #f59e0b; }
    .source-card-low  { border-left: 3px solid #ef4444; }
    .metric-box { background: linear-gradient(135deg, #1e293b, #111827); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .example-btn { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 0.75rem 1rem; cursor: pointer; margin: 0.25rem 0; transition: all 0.2s; }
    .example-btn:hover { background: rgba(99,102,241,0.15); border-color: rgba(99,102,241,0.3); }
    .streaming-indicator { display: inline-block; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; animation: pulse 1.5s infinite; margin-right: 6px; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
""", unsafe_allow_html=True)

# ── Cached engine ─────────────────────────────────────────────────
@st.cache_resource
def load_engine() -> QAEngine:
    return QAEngine(searcher=Searcher(), top_k=5)


engine = load_engine()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ DocQ")
    st.caption("Production RAG QA Engine")
    st.divider()

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        docs = len({m["source"] for m in engine.searcher.collection.get()["metadatas"] if m})
        st.metric("Docs", docs)
    with m2:
        st.metric("Chunks", engine.searcher.collection.count())
    with m3:
        st.metric("Model", "3B")

    st.divider()

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Retrieved chunks", 1, 10, 5, help="More chunks = more context, slower response")
    engine.top_k = top_k

    st.divider()

    st.markdown("### 📄 Indexed Documents")
    sources = [m["source"] for m in engine.searcher.collection.get()["metadatas"] if m]
    for s in sorted(set(sources)):
        st.markdown(f"• `{s}`")

    st.divider()

    with st.expander("🔬 Evaluation Results", expanded=False):
        st.markdown("""
        | Metric | Score |
        |--------|-------|
        | Hit Rate@5 | **100%** |
        | MRR | **0.920** |
        | Avg Relevance | **5.0/5** |
        | Avg Groundedness | **5.0/5** |
        """)

    st.divider()
    st.caption("Llama 3.2 3B + ChromaDB + FastAPI")

# ── Main ──────────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:2rem; margin-bottom:0;">Ask your documents anything</h1>', unsafe_allow_html=True)
st.caption("RAG-powered answers with source citations. All running locally — no API keys, no cloud.")


# ── Example questions ─────────────────────────────────────────────
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Try asking:")
    examples = [
        "What is RAG and how does it work?",
        "Explain the transformer attention mechanism",
        "How do I handle async programming in Python?",
        "What are the best prompt engineering techniques?",
        "Compare cosine similarity vs Euclidean distance",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.messages = [{"role": "user", "content": ex}]
                st.rerun()


# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f'<div class="chat-assistant"></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-user"></div>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📖 View sources"):
                for s in msg["sources"]:
                    score = s["score"]
                    if score > 0.3:
                        card_class = "source-card source-card-high"
                        emoji = "🟢"
                    elif score > 0.0:
                        card_class = "source-card source-card-mid"
                        emoji = "🟡"
                    else:
                        card_class = "source-card source-card-low"
                        emoji = "🔴"
                    st.markdown(
                        f'<div class="{card_class}"><strong>{emoji} {s["document"]}</strong> '
                        f'<span style="opacity:0.6">score: {s["score"]:.3f}</span><br><small>{s["preview"]}</small></div>',
                        unsafe_allow_html=True,
                    )

# ── Chat input ────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        spinner = st.empty()
        spinner.markdown('<div class="streaming-indicator"></div> Thinking...', unsafe_allow_html=True)

        tokens = []
        sources_data = None

        for item in engine.answer_stream(prompt):
            if isinstance(item, dict) and item.get("_done"):
                sources_data = item
            else:
                tokens.append(item)
                placeholder.markdown("".join(tokens) + "▌")

        # Final render
        spinner.empty()
        full = sources_data["full_answer"] if sources_data else "".join(tokens)
        placeholder.markdown(full)

        # Format sources for history
        source_list = []
        if sources_data:
            for i, s in enumerate(sources_data["sources"]):
                preview = sources_data["context_chunks"][i][:120] + "..."
                source_list.append({
                    "document": s["document"],
                    "score": s["score"],
                    "preview": preview,
                })

        st.session_state.messages.append({
            "role": "assistant",
            "content": full,
            "sources": source_list,
        })
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("⚡ DocQ RAG Engine · Llama 3.2 3B · ChromaDB · all-MiniLM-L6-v2 · All local inference")
