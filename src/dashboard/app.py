import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
from collections import Counter
from src.storage.vector_store import VectorStore
from src.processing.embedder import Embedder
from config import TARGET_COMPANY

st.set_page_config(
    page_title=f"{TARGET_COMPANY} CEO Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title(f"🧠 {TARGET_COMPANY} Strategic Intelligence Dashboard")
st.caption("AI-powered CEO briefing tool — powered by RAG + Qwen LLM")


@st.cache_resource
def load_store():
    return VectorStore()


@st.cache_resource
def load_embedder():
    return Embedder()


@st.cache_resource
def load_agent():
    from src.agent.ceo_agent import CEOAgent
    return CEOAgent()


store = load_store()
embedder = load_embedder()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Knowledge Base")
    total = store.count()
    st.metric("Chunks stored", total)

    st.divider()
    st.header("Run Pipeline")
    if st.button("Collect + Process + Store", type="primary"):
        with st.spinner("Running full pipeline..."):
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
            added = store.store(embedded)
            st.success(f"Pipeline complete! {added} new chunks stored.")
            st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Ask the Agent", "Sentiment Overview", "Topic Breakdown"])

# Tab 1 — RAG Q&A
with tab1:
    st.subheader("Ask a Strategic Question")

    suggested = [
        f"What are the biggest risks facing {TARGET_COMPANY} right now?",
        f"What is the market sentiment around {TARGET_COMPANY}?",
        f"What new products has {TARGET_COMPANY} announced recently?",
        f"How is {TARGET_COMPANY} positioned against its competitors?",
    ]

    with st.expander("Suggested questions"):
        for q in suggested:
            if st.button(q, key=q):
                st.session_state["question"] = q

    question = st.text_input(
        "Your question:",
        value=st.session_state.get("question", ""),
        placeholder=f"e.g. What is the outlook for {TARGET_COMPANY} stock?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving context and generating answer..."):
            agent = load_agent()
            result = agent.query(question)

        st.markdown("### Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.markdown("### Sources used")
            for s in result["sources"]:
                with st.expander(f"[{s['source']}] {s['date']} — {s['text'][:80]}..."):
                    st.write(s["text"])
                    if s["url"]:
                        st.markdown(f"[Read more]({s['url']})")

# Tab 2 — Sentiment
with tab2:
    st.subheader("Sentiment Distribution")

    if total == 0:
        st.info("No data yet. Run the pipeline from the sidebar.")
    else:
        results = store.collection.get(include=["metadatas"])
        sentiments = [m.get("sentiment", "unknown") for m in results["metadatas"]]
        counts = Counter(sentiments)

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", counts.get("positive", 0))
        col2.metric("Neutral", counts.get("neutral", 0))
        col3.metric("Negative", counts.get("negative", 0))

        import pandas as pd
        df = pd.DataFrame(counts.items(), columns=["Sentiment", "Count"])
        st.bar_chart(df.set_index("Sentiment"))

# Tab 3 — Topics
with tab3:
    st.subheader("Topic Distribution")

    if total == 0:
        st.info("No data yet. Run the pipeline from the sidebar.")
    else:
        results = store.collection.get(include=["metadatas"])
        topics = [m.get("topic", "unknown") for m in results["metadatas"]]
        topic_counts = Counter(topics)

        import pandas as pd
        df = pd.DataFrame(topic_counts.most_common(), columns=["Topic", "Count"])
        st.bar_chart(df.set_index("Topic"))
        st.dataframe(df, use_container_width=True)
