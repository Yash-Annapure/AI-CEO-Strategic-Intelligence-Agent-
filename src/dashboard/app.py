import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
from collections import Counter
import pandas as pd
import altair as alt
from src.storage.vector_store import VectorStore
from config import TARGET_COMPANY

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "results", "dashboard_data.json"
)

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


def load_dashboard_data():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def render_agent_section(data: dict):
    """renders a pre-computed agent result — tool trace + answer."""
    if not data:
        st.info("No data yet — run the pipeline from the sidebar.")
        return
    if data.get("tool_calls"):
        with st.expander("Agent trace — tools called", expanded=False):
            for tc in data["tool_calls"]:
                args_str = ", ".join(
                    f'{k}="{v}"' for k, v in tc["input"].items()
                ) if tc["input"] else ""
                st.code(f"{tc['tool']}({args_str})", language=None)
    st.write(data.get("answer", "No answer generated."))


store = load_store()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    ollama_model = st.text_input("Ollama model", value="qwen2.5:3b")
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
            import datetime

            raw = collect_reddit() + collect_news() + collect_rss()
            cleaned = clean_all(raw)
            cleaned = analyze_all(cleaned)
            cleaned = classify_all(cleaned)
            chunks = chunk_all(cleaned)
            emb = Embedder()
            embedded = emb.embed_chunks(chunks)
            store.clear()
            added = store.store(embedded)
            st.success(f"Data stored: {added} chunks.")

        with st.spinner("Running strategic agent analysis (4 sections)..."):
            from src.agent.langgraph_agent import run_agent
            from main import AGENT_GOALS
            import datetime
            try:
                os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
                dash = {"generated_at": datetime.datetime.now().isoformat()}
                for key, goal in AGENT_GOALS.items():
                    dash[key] = run_agent(goal, ollama_model=ollama_model)
                with open(RESULTS_PATH, "w") as f:
                    json.dump(dash, f, indent=2)
                st.success("Agent analysis complete.")
            except Exception as e:
                st.warning(f"Agent step failed: {e}")

        st.cache_resource.clear()
        st.rerun()

# ── Fetch data once ───────────────────────────────────────────────────────────
total_chunks = store.count()
all_meta = store.collection.get(include=["metadatas"])["metadatas"] if total_chunks > 0 else []
sources = list(set(m.get("source", "") for m in all_meta if m.get("source")))
dash_data = load_dashboard_data()

# ── Section 1: Company Overview ───────────────────────────────────────────────
st.header("Section 1: Company Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Company", TARGET_COMPANY)
c2.metric("Industry", "Semiconductors & AI")
c3.metric("Documents collected", total_chunks)
c4.metric("Data sources", len(sources))
st.divider()

# ── Section 2: Market Intelligence ───────────────────────────────────────────
st.header("Section 2: Market Intelligence")
if total_chunks == 0:
    st.info("No data yet. Run the pipeline from the sidebar.")
else:
    all_docs = store.collection.get(include=["metadatas", "documents"])
    all_paired = list(zip(all_docs["metadatas"], all_docs["documents"]))

    news_items = sorted(
        [(m, d) for m, d in all_paired if "reddit" not in m.get("source", "").lower()],
        key=lambda x: x[0].get("date", ""),
        reverse=True,
    )[:6]

    reddit_items = sorted(
        [(m, d) for m, d in all_paired if "reddit" in m.get("source", "").lower()],
        key=lambda x: x[0].get("date", ""),
        reverse=True,
    )[:6]

    col_news, col_reddit = st.columns(2)
    with col_news:
        st.subheader("Recent News & Announcements")
        for meta, doc in news_items[:6]:
            src = meta.get("source", "")
            date = meta.get("date", "")[:10]
            title = doc.split(".")[0][:90]
            url = meta.get("url", "")
            topic = meta.get("topic", "")
            st.markdown(f"- **[{src}]** `{topic}` {date} — [{title}]({url})")
        if not news_items:
            st.info("No news articles collected.")

    with col_reddit:
        st.subheader("Public Sentiment & Discussions")
        for meta, doc in reddit_items[:6]:
            src = meta.get("source", "")
            date = meta.get("date", "")[:10]
            title = doc.split(".")[0][:90]
            url = meta.get("url", "")
            sent = meta.get("sentiment", "")
            sent_badge = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(sent, "⚪")
            st.markdown(f"- {sent_badge} **[{src}]** {date} — [{title}]({url})")
        if not reddit_items:
            st.info("No Reddit discussions collected.")
st.divider()

# ── Section 3: Opportunity Monitor ───────────────────────────────────────────
st.header("Section 3: Opportunity Monitor")
render_agent_section(dash_data.get("opportunities") if dash_data else None)
st.divider()

# ── Section 4: Risk Monitor ───────────────────────────────────────────────────
st.header("Section 4: Risk Monitor")
render_agent_section(dash_data.get("risks") if dash_data else None)
st.divider()

# ── Section 5: Sentiment Analysis ─────────────────────────────────────────────
st.header("Section 5: Sentiment Analysis")
if total_chunks == 0:
    st.info("No data yet. Run the pipeline from the sidebar.")
else:
    news_meta = [m for m in all_meta if "reddit" not in m.get("source", "").lower()]
    reddit_meta = [m for m in all_meta if "reddit" in m.get("source", "").lower()]

    news_sent = Counter(m.get("sentiment", "unknown") for m in news_meta)
    reddit_sent = Counter(m.get("sentiment", "unknown") for m in reddit_meta)

    col_left, col_right = st.columns(2)

    def _sentiment_chart(df):
        sort_order = ["positive", "neutral", "negative"]
        base = alt.Chart(df).encode(
            x=alt.X("Sentiment:N", title=None,
                    axis=alt.Axis(labelAngle=0),
                    sort=sort_order),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["#2ECC71", "#95A5A6", "#E74C3C"],
                ),
                legend=None,
            ),
        )
        bars = base.mark_bar().encode(y=alt.Y("Count:Q", title="Count"))
        labels = base.mark_text(align="center", baseline="bottom", dy=-4, color="white").encode(
            y=alt.Y("Count:Q"),
            text=alt.Text("Count:Q"),
        )
        return (bars + labels).properties(height=300)

    with col_left:
        st.subheader("News Sentiment")
        if news_meta:
            df_news = pd.DataFrame([
                {"Sentiment": k, "Count": v} for k, v in news_sent.items()
            ])
            st.altair_chart(_sentiment_chart(df_news), use_container_width=True)
        else:
            st.info("No news articles in knowledge base.")

    with col_right:
        st.subheader("Public Sentiment (Reddit)")
        if reddit_meta:
            df_reddit = pd.DataFrame([
                {"Sentiment": k, "Count": v} for k, v in reddit_sent.items()
            ])
            st.altair_chart(_sentiment_chart(df_reddit), use_container_width=True)
        else:
            st.info("No Reddit data in knowledge base.")

    st.subheader("Sentiment Trends — Topic Breakdown")
    topics = [m.get("topic", "unknown") for m in all_meta]
    topic_counts = Counter(topics)
    df_topics = pd.DataFrame(topic_counts.most_common(), columns=["Topic", "Count"])
    topic_chart = (
        alt.Chart(df_topics).mark_bar()
        .encode(
            x=alt.X("Count:Q", title="Count"),
            y=alt.Y("Topic:N", sort="-x", title=None),
            color=alt.value("#4C9BE8"),
        )
        .properties(height=300)
    )
    st.altair_chart(topic_chart, use_container_width=True)
st.divider()

# ── Section 6: Strategic Recommendations ─────────────────────────────────────
st.header("Section 6: Strategic Recommendations")
render_agent_section(dash_data.get("recommendations") if dash_data else None)
st.divider()

# ── Section 7: CEO Briefing ───────────────────────────────────────────────────
st.header("Section 7: CEO Briefing")
if dash_data and dash_data.get("ceo_briefing"):
    render_agent_section(dash_data["ceo_briefing"])
    st.divider()
    st.subheader("Ask the Strategic Agent")
else:
    if total_chunks == 0:
        st.info("No data yet. Run the pipeline first.")
    else:
        st.info("Run the full pipeline to generate the CEO briefing.")

if total_chunks > 0:
    st.write(
        "Ask any strategic question. The agent autonomously decides which tools "
        "to call, retrieves evidence from the knowledge base, and produces a "
        "grounded recommendation."
    )
    goal = st.text_input(
        "Strategic goal:",
        placeholder=f"e.g. How should {TARGET_COMPANY} respond to rising competition from AMD?",
    )
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
