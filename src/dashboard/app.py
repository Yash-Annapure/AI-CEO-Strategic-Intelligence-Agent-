import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from collections import Counter
import pandas as pd
import altair as alt
from src.storage.vector_store import VectorStore
from config import TARGET_COMPANY

st.set_page_config(
    page_title=f"{TARGET_COMPANY} CEO Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
)

col_logo, col_title = st.columns([1, 7])
with col_logo:
    _logo_path = os.path.join(os.path.dirname(__file__), "assets", "NVIDIA_LOGO.png")
    with open(_logo_path, "rb") as _f:
        st.image(_f.read(), width=130)
with col_title:
    st.title(f"{TARGET_COMPANY} Strategic Intelligence Dashboard")


@st.cache_resource
def load_store():
    return VectorStore()


store = load_store()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("Run Full Pipeline", type="primary"):
        with st.spinner("Collecting and processing data..."):
            from src.collectors.reddit_collector import collect as collect_reddit
            from src.collectors.news_collector import collect as collect_news
            from src.collectors.rss_collector import collect as collect_rss
            from src.processing.cleaner import clean_all
            from src.processing.chunker import chunk_all
            from src.processing.embedder import Embedder
            from src.intelligence.sentiment import analyze_all
            from src.intelligence.classifier import classify_all

            raw = collect_reddit() + collect_news() + collect_rss()
            cleaned = clean_all(raw)
            cleaned = analyze_all(cleaned)
            cleaned = classify_all(cleaned)
            chunks = chunk_all(cleaned)
            emb = Embedder()
            embedded = emb.embed_chunks(chunks)
            store.clear()
            added = store.store(embedded)
            st.success(f"Done! {added} chunks stored.")
            st.cache_resource.clear()
            st.rerun()

# ── Fetch metadata once ───────────────────────────────────────────────────────
total_chunks = store.count()
all_meta = store.collection.get(include=["metadatas"])["metadatas"] if total_chunks > 0 else []
sources = list(set(m.get("source", "") for m in all_meta if m.get("source")))

# ── Section 1: System Overview ────────────────────────────────────────────────
st.header("Section 1: System Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Company", TARGET_COMPANY)
col2.metric("Industry", "Semis & AI")
col3.metric("Chunks in knowledge base", total_chunks)
col4.metric("Data sources", len(sources))
st.divider()

# ── Section 2: Intelligence Feed ──────────────────────────────────────────────
st.header("Section 2: Intelligence Feed")

if total_chunks == 0:
    st.info("No data yet. Run the pipeline from the sidebar.")
else:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Recent Articles")
        recent_docs = store.collection.get(include=["metadatas", "documents"])
        paired = sorted(
            zip(recent_docs["metadatas"], recent_docs["documents"]),
            key=lambda x: x[0].get("date", ""),
            reverse=True,
        )[:8]
        for meta, doc in paired:
            src = meta.get("source", "")
            date = meta.get("date", "")[:10]
            title = doc.split(".")[0][:90]
            url = meta.get("url", "")
            st.markdown(f"- **[{src}]** {date} — [{title}]({url})")

    with col_right:
        st.subheader("Topic Breakdown")
        topics = [m.get("topic", "unknown") for m in all_meta]
        topic_counts = Counter(topics)
        df_topics = pd.DataFrame(topic_counts.most_common(), columns=["Topic", "Count"])
        topic_chart = (
            alt.Chart(df_topics)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Topic:N", sort="-x", title=None),
                color=alt.value("#4C9BE8"),
            )
            .properties(height=300)
        )
        st.altair_chart(topic_chart, use_container_width=True)

    st.subheader("Sentiment Breakdown")
    sentiments = [m.get("sentiment", "unknown") for m in all_meta]
    sent_counts = Counter(sentiments)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Positive", sent_counts.get("positive", 0))
    col_b.metric("Neutral", sent_counts.get("neutral", 0))
    col_c.metric("Negative", sent_counts.get("negative", 0))

    df_sent = pd.DataFrame([{"Sentiment": k, "Count": v} for k, v in sent_counts.items()])
    sent_chart = (
        alt.Chart(df_sent)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q"),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["#2ECC71", "#95A5A6", "#E74C3C"],
                ),
                legend=None,
            ),
        )
        .properties(height=250)
    )
    st.altair_chart(sent_chart, use_container_width=True)

st.divider()

# ── Section 3: Strategic Agent ────────────────────────────────────────────────
st.header("Section 3: Strategic Agent")

if total_chunks == 0:
    st.info("No data yet. Run the pipeline first.")
else:
    st.write(
        "Enter a strategic goal. The agent will autonomously decide which tools to call, "
        "retrieve evidence from the knowledge base, and produce a grounded recommendation."
    )

    goal = st.text_input(
        "Strategic goal:",
        placeholder=f"e.g. What are the biggest risks facing {TARGET_COMPANY} right now?",
    )

    ollama_model = st.sidebar.text_input("Ollama model", value="qwen2.5:3b")

    if st.button("Run Agent", type="primary") and goal:
        with st.spinner("Agent is running..."):
            from src.agent.langgraph_agent import run_agent
            result = run_agent(goal, ollama_model=ollama_model)
            st.session_state["agent_result"] = result

    if "agent_result" in st.session_state:
        res = st.session_state["agent_result"]

        if res["tool_calls"]:
            with st.expander("Agent trace — tools called", expanded=True):
                for tc in res["tool_calls"]:
                    args_str = ", ".join(
                        f'{k}="{v}"' for k, v in tc["input"].items()
                    ) if tc["input"] else ""
                    st.code(f"{tc['tool']}({args_str})", language=None)

        st.subheader("Agent Answer")
        st.write(res["answer"])
