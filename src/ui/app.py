"""Streamlit UI — multi-model RAG with Groq 70B (default), Groq 8B, Llama local."""

import os, sys
from pathlib import Path
import streamlit as st

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from retrieval.qa_engine import QAEngine, MODELS
from retrieval.searcher import Searcher

st.set_page_config(
    page_title="DocQ — RAG QA Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ─────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #090a0f 0%, #11131f 30%, #141828 100%);
    }
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, sans-serif;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0c14 0%, #0f111d 100%);
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] .stMetric {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
    }
    section[data-testid="stSidebar"] .stMetric label {
        font-size: 0.65rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.35) !important;
    }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }

    /* ── Sidebar source list ──────────────────────────────── */
    .sidebar-source {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.45);
        padding: 2px 6px;
        border-radius: 4px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 2px;
    }

    /* ── Chat bubbles ─────────────────────────────────────── */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(37,99,235,0.08));
        border: 1px solid rgba(124,58,237,0.15);
        border-radius: 16px 16px 4px 16px;
        padding: 0.9rem 1.2rem;
        margin: 0.3rem 0;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .chat-bubble-assistant {
        background: transparent;
        border-left: 2px solid rgba(255,255,255,0.05);
        padding: 0.2rem 0 0.2rem 1.2rem;
        margin: 0.6rem 0;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* ── Source cards ─────────────────────────────────────── */
    .source-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.7), rgba(17,24,39,0.7));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        position: relative;
        overflow: hidden;
    }
    .source-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        border-radius: 3px 0 0 3px;
    }
    .source-card-high::before { background: #22c55e; }
    .source-card-mid::before  { background: #f59e0b; }
    .source-card-low::before  { background: #ef4444; }
    .source-card .score-bar {
        height: 3px;
        border-radius: 3px;
        margin-top: 6px;
        background: rgba(255,255,255,0.05);
    }
    .source-card .score-bar-fill {
        height: 100%;
        border-radius: 3px;
    }
    .source-card .doc-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .source-card .doc-score {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.35);
        font-family: 'JetBrains Mono', monospace;
    }
    .source-card .doc-preview {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
        line-height: 1.45;
        margin-top: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* ── Model badges ─────────────────────────────────────── */
    .model-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .model-cloud {
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(37,99,235,0.12));
        border: 1px solid rgba(124,58,237,0.35);
        color: #a78bfa;
    }
    .model-local {
        background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(22,163,74,0.12));
        border: 1px solid rgba(34,197,94,0.35);
        color: #4ade80;
    }
    .model-selector {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.3rem 0;
    }

    /* ── Example questions ────────────────────────────────── */
    .example-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 0.5rem 0 1rem 0;
    }
    .example-pill-btn {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.55);
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
    }
    .example-pill-btn:hover {
        background: rgba(124,58,237,0.15);
        border-color: rgba(124,58,237,0.3);
        color: #c4b5fd;
    }

    /* ── Thinking indicator ──────────────────────────────── */
    .thinking-box {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0;
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
    }
    .thinking-dots { display: flex; gap: 3px; }
    .thinking-dots span {
        width: 5px; height: 5px;
        background: #a78bfa;
        border-radius: 50%;
        animation: dotPulse 1.4s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes dotPulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.3; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* ── Streaming cursor ─────────────────────────────────── */
    .streaming-text .cursor {
        display: inline-block;
        width: 2px;
        height: 1em;
        background: #a78bfa;
        margin-left: 2px;
        animation: blink 1s step-end infinite;
        vertical-align: text-bottom;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* ── Sidebar section header ───────────────────────────── */
    .sidebar-heading {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255,255,255,0.3);
        margin: 1rem 0 0.4rem 0;
    }

    /* ── Misc ─────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.05) !important; }
    .stExpander { border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 10px !important; }
    .stExpander header { font-size: 0.82rem !important; }
    .stCaption { color: rgba(255,255,255,0.3) !important; }
    .emoji-avatar { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────
DEFAULT_MODEL = "groq-70b"

if "engine" not in st.session_state:
    st.session_state.engine = QAEngine(
        searcher=Searcher(),
        model_key=DEFAULT_MODEL,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

engine = st.session_state.engine

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ DocQ")
    st.caption("Multi-Model RAG QA Engine")

    # ── Model section ──
    st.divider()
    options = list(MODELS.keys())
    labels = []
    for k in options:
        cfg = MODELS[k]
        emoji = "☁️" if cfg["provider"] != "ollama" else "💻"
        labels.append(f"{emoji} {cfg['name']}")

    selected_label = st.selectbox(
        "Select LLM",
        options=labels,
        index=options.index(engine.model_key),
        key="model_select",
        label_visibility="collapsed",
    )
    selected_key = options[labels.index(selected_label)]
    if selected_key != engine.model_key:
        engine.switch_model(selected_key)
        st.toast(f"Switched to {MODELS[selected_key]['name']}")

    cfg = MODELS[engine.model_key]
    badge_class = "model-cloud" if cfg["provider"] != "ollama" else "model-local"
    badge_label = "☁️ Cloud · Groq" if cfg["provider"] != "ollama" else "💻 Local · Ollama"
    st.markdown(
        f'<span class="model-badge {badge_class}">{badge_label}</span>',
        unsafe_allow_html=True,
    )

    # ── Index stats ──
    st.markdown('<div class="sidebar-heading">Index</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        docs = len({m["source"] for m in engine.searcher.collection.get()["metadatas"] if m})
        st.metric("Documents", docs)
    with m2:
        st.metric("Chunks", engine.searcher.collection.count())

    # ── Retrieval settings ──
    st.markdown('<div class="sidebar-heading">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks", 1, 10, 5, label_visibility="collapsed")
    engine.top_k = top_k

    # ── Sources ──
    st.markdown('<div class="sidebar-heading">Indexed Sources</div>', unsafe_allow_html=True)
    sources = [m["source"] for m in engine.searcher.collection.get()["metadatas"] if m]
    for s in sorted(set(sources)):
        st.markdown(f'<div class="sidebar-source">📄 {s}</div>', unsafe_allow_html=True)

    # ── Eval ──
    st.markdown('<div class="sidebar-heading">Benchmarks</div>', unsafe_allow_html=True)
    with st.expander("🔬 View results", expanded=False):
        st.markdown("""
        | Metric | Score |
        |--------|-------|
        | Hit Rate@5 | **100%** |
        | MRR | **0.920** |
        """)

    st.divider()
    st.caption("Groq 70B · ChromaDB · FastAPI")

# ── Header ────────────────────────────────────────────────────────
cfg = MODELS[engine.model_key]
badge_class = "model-cloud" if cfg["provider"] != "ollama" else "model-local"
emoji = "☁️" if cfg["provider"] != "ollama" else "💻"
st.markdown(
    f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px;">'
    f'<span style="font-size:1.6rem;font-weight:600;color:#f1f5f9;">Ask your documents</span>'
    f'<span class="model-badge {badge_class}">{emoji} {engine.model_name}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.caption("RAG-powered answers with source citations · 70B via Groq (free tier)")

# ── Example questions ─────────────────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Try a question")
    examples = [
        "What is RAG and how does it work?",
        "Explain the transformer attention mechanism",
        "How do I handle async programming in Python?",
        "What are the best prompt engineering techniques?",
        "Compare cosine similarity vs Euclidean distance",
    ]

    # 2-row grid: 3 cols then 2 cols
    cols_per_row = 3
    for row_start in range(0, len(examples), cols_per_row):
        row_examples = examples[row_start:row_start + cols_per_row]
        row_cols = st.columns(len(row_examples))
        for i, ex in enumerate(row_examples):
            with row_cols[i]:
                if st.button(ex, key=f"ex_{row_start + i}", use_container_width=True):
                    st.session_state.messages = [{"role": "user", "content": ex}]
                    st.rerun()

# ── Chat messages ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-assistant">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Source cards with score bars, scrollable
            if msg.get("sources"):
                source_html = '<div style="margin-top:0.8rem;">'
                source_html += '<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.25);margin-bottom:0.3rem;">📖 Sources</div>'
                source_html += '<div style="max-height:280px;overflow-y:auto;padding-right:4px;">'
                for s in msg["sources"]:
                    score = s["score"]
                    if score > 0.3:
                        card_class = "source-card source-card-high"
                        bar_color = "#22c55e"
                        bar_width = f"{min(score * 100, 100)}%"
                    elif score > 0.0:
                        card_class = "source-card source-card-mid"
                        bar_color = "#f59e0b"
                        bar_width = f"{min(score * 100, 100)}%"
                    else:
                        card_class = "source-card source-card-low"
                        bar_color = "#ef4444"
                        bar_width = f"{max(min(abs(score) * 100, 100), 5)}%"

                    source_html += (
                        f'<div class="{card_class}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span class="doc-title">{s["document"]}</span>'
                        f'<span class="doc-score">{s["score"]:.3f}</span>'
                        f'</div>'
                        f'<div class="doc-preview">{s["preview"]}</div>'
                        f'<div class="score-bar"><div class="score-bar-fill" style="width:{bar_width};background:{bar_color};"></div></div>'
                        f'</div>'
                    )
                source_html += '</div></div>'
                st.markdown(source_html, unsafe_allow_html=True)

# ── Chat input & streaming ────────────────────────────────────────
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(
            f'<div class="chat-bubble-user">{prompt}</div>',
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant", avatar="⚡"):
        placeholder = st.empty()
        thinking_box = st.empty()

        thinking_box.markdown(
            '<div class="thinking-box">'
            '<div class="thinking-dots"><span></span><span></span><span></span></div>'
            'Thinking...'
            '</div>',
            unsafe_allow_html=True,
        )

        tokens = []
        meta = None

        for item in engine.answer_stream(prompt):
            if isinstance(item, dict) and item.get("_done"):
                meta = item
            else:
                tokens.append(item)
                rendered = "".join(tokens)
                placeholder.markdown(
                    f'<div class="chat-bubble-assistant streaming-text">'
                    f'{rendered}<span class="cursor"></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        thinking_box.empty()
        full = meta["full_answer"] if meta else "".join(tokens)

        source_cards = []
        source_html = ""
        if meta and meta.get("sources"):
            for i, s in enumerate(meta["sources"]):
                preview = meta["context_chunks"][i][:120] + "..."
                source_cards.append({"document": s["document"], "score": s["score"], "preview": preview})

                score = s["score"]
                if score > 0.3:
                    card_class = "source-card source-card-high"
                    bar_color = "#22c55e"
                    bar_width = f"{min(score * 100, 100)}%"
                elif score > 0.0:
                    card_class = "source-card source-card-mid"
                    bar_color = "#f59e0b"
                    bar_width = f"{min(score * 100, 100)}%"
                else:
                    card_class = "source-card source-card-low"
                    bar_color = "#ef4444"
                    bar_width = f"{max(min(abs(score) * 100, 100), 5)}%"

                source_html += (
                    f'<div class="{card_class}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span class="doc-title">{s["document"]}</span>'
                    f'<span class="doc-score">{s["score"]:.3f}</span>'
                    f'</div>'
                    f'<div class="doc-preview">{preview}</div>'
                    f'<div class="score-bar"><div class="score-bar-fill" style="width:{bar_width};background:{bar_color};"></div></div>'
                    f'</div>'
                )

        final_html = f'<div class="chat-bubble-assistant">{full}</div>'
        if source_html:
            final_html += (
                '<div style="margin-top:0.8rem;">'
                '<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.25);margin-bottom:0.3rem;">📖 Sources</div>'
                f'<div style="max-height:280px;overflow-y:auto;padding-right:4px;">{source_html}</div>'
                '</div>'
            )
        placeholder.markdown(final_html, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full,
            "sources": source_cards,
        })
        st.rerun()

st.divider()
st.caption("⚡ DocQ RAG Engine · Groq 70B · ChromaDB · FastAPI · Streamlit")
