import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from collections import Counter
from datetime import datetime
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
    st.image(
        os.path.join(os.path.dirname(__file__), "assets", "nvidia_logo.png"),
        width=130,
    )
with col_title:
    st.title(f"{TARGET_COMPANY} Strategic Intelligence Dashboard")


@st.cache_resource
def load_store():
    return VectorStore()


@st.cache_resource
def load_agent():
    from src.agent.ceo_agent import CEOAgent
    return CEOAgent()


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
            added = store.store(embedded)
            st.success(f"Done! {added} new chunks stored.")
            st.cache_resource.clear()
            st.rerun()

# ── Fetch metadata once ───────────────────────────────────────────────────────
total_chunks = store.count()
all_meta = store.collection.get(include=["metadatas"])["metadatas"] if total_chunks > 0 else []
sources = list(set(m.get("source", "") for m in all_meta if m.get("source")))
dates = sorted([m.get("date", "")[:10] for m in all_meta if m.get("date")], reverse=True)
last_update = dates[0] if dates else "No data yet"

# ── Section 1: Company Overview ───────────────────────────────────────────────
st.header("Section 1: Company Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Company", TARGET_COMPANY)
col2.metric("Industry", "Semis & AI")
col3.metric("Documents", total_chunks)
col4.metric("Data sources", len(sources))
col5.metric("Last updated", last_update)
st.divider()

# ── Section 2: Market Intelligence ───────────────────────────────────────────
st.header("Section 2: Market Intelligence")

if total_chunks == 0:
    st.info("No data yet. Run the pipeline from the sidebar.")
else:
    topics = [m.get("topic", "unknown") for m in all_meta]
    topic_counts = Counter(topics)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Recent News")
        recent_docs = store.collection.get(include=["metadatas", "documents"])
        paired = sorted(
            zip(recent_docs["metadatas"], recent_docs["documents"]),
            key=lambda x: x[0].get("date", ""),
            reverse=True
        )[:8]
        for meta, doc in paired:
            src = meta.get("source", "")
            date = meta.get("date", "")[:10]
            title = doc.split(".")[0][:90]
            url = meta.get("url", "")
            st.markdown(f"- **[{src}]** {date} — [{title}]({url})")

    with col_b:
        st.subheader("Topic Breakdown")
        df_topics = pd.DataFrame(topic_counts.most_common(), columns=["Topic", "Count"])
        chart = (
            alt.Chart(df_topics)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Topic:N", sort="-x", title=None),
                color=alt.value("#4C9BE8"),
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Emerging Technology Topics")
    ai_topics = [t for t in topics if "AI" in t or "machine" in t.lower() or "product" in t.lower()]
    st.write(f"AI/Technology-related chunks: **{len(ai_topics)}** out of {total_chunks}")

st.divider()

# ── Section 3: Opportunity Monitor ───────────────────────────────────────────
st.header("Section 3: Opportunity Monitor")

if total_chunks == 0:
    st.info("No data yet.")
else:
    if st.button("Identify Opportunities", key="opp_btn"):
        with st.spinner("Analyzing opportunities..."):
            agent = load_agent()
            opportunities = agent.analyze_opportunities()
            st.session_state["opportunities"] = opportunities

    if "opportunities" in st.session_state:
        opps = st.session_state["opportunities"]
        if opps:
            for opp in opps:
                impact = opp.get("impact", "Medium")
                color = "🟢" if impact == "High" else "🟡" if impact == "Medium" else "🔴"
                with st.expander(f"{color} {opp.get('title', 'Opportunity')} — Impact: {impact}"):
                    st.write(f"**Evidence:** {opp.get('evidence', 'N/A')}")
                    st.write(f"**Confidence:** {opp.get('confidence', 'N/A')}")
        else:
            st.warning("Could not parse opportunities. Try again.")

st.divider()

# ── Section 4: Risk Monitor ───────────────────────────────────────────────────
st.header("Section 4: Risk Monitor")

if total_chunks == 0:
    st.info("No data yet.")
else:
    if st.button("Identify Risks", key="risk_btn"):
        with st.spinner("Analyzing risks..."):
            agent = load_agent()
            risks = agent.analyze_risks()
            st.session_state["risks"] = risks

    if "risks" in st.session_state:
        risks = st.session_state["risks"]
        if risks:
            for risk in risks:
                severity = risk.get("severity", "Medium")
                color = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                with st.expander(f"{color} {risk.get('title', 'Risk')} — Severity: {severity}"):
                    st.write(f"**Category:** {risk.get('category', 'N/A')}")
                    st.write(f"**Evidence:** {risk.get('evidence', 'N/A')}")
                    st.write(f"**Confidence:** {risk.get('confidence', 'N/A')}")
        else:
            st.warning("Could not parse risks. Try again.")

st.divider()

# ── Section 5: Sentiment Analysis ────────────────────────────────────────────
st.header("Section 5: Sentiment Analysis")

if total_chunks == 0:
    st.info("No data yet.")
else:
    sentiments = [m.get("sentiment", "unknown") for m in all_meta]
    counts = Counter(sentiments)

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", counts.get("positive", 0))
    col2.metric("Neutral", counts.get("neutral", 0))
    col3.metric("Negative", counts.get("negative", 0))

    df_sent = pd.DataFrame(
        [{"Sentiment": k, "Count": v} for k, v in counts.items()]
    )
    sent_chart = (
        alt.Chart(df_sent)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["#2ECC71", "#95A5A6", "#E74C3C"],
                ),
                legend=None,
            ),
        )
        .properties(height=280)
    )
    st.altair_chart(sent_chart, use_container_width=True)

    # news vs public (Reddit) sentiment split
    news_sentiments = [m.get("sentiment") for m in all_meta if "reddit" not in m.get("source", "").lower()]
    reddit_sentiments = [m.get("sentiment") for m in all_meta if "reddit" in m.get("source", "").lower()]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("News Sentiment")
        nc = Counter(news_sentiments)
        st.write(f"Positive: {nc.get('positive',0)} | Neutral: {nc.get('neutral',0)} | Negative: {nc.get('negative',0)}")
    with col_b:
        st.subheader("Public Sentiment (Reddit)")
        rc = Counter(reddit_sentiments)
        st.write(f"Positive: {rc.get('positive',0)} | Neutral: {rc.get('neutral',0)} | Negative: {rc.get('negative',0)}")

st.divider()

# ── Section 6: Strategic Recommendations ─────────────────────────────────────
st.header("Section 6: Strategic Recommendations")

if total_chunks == 0:
    st.info("No data yet.")
else:
    if st.button("Generate Recommendations", key="rec_btn"):
        with st.spinner("Generating strategic recommendations..."):
            agent = load_agent()
            recs = agent.generate_recommendations()
            st.session_state["recommendations"] = recs

    if "recommendations" in st.session_state:
        recs = st.session_state["recommendations"]
        if recs:
            for i, rec in enumerate(recs, 1):
                priority = rec.get("priority", "Medium")
                badge = "🔴 HIGH" if priority == "High" else "🟡 MEDIUM" if priority == "Medium" else "🟢 LOW"
                with st.expander(f"Recommendation {i}: {rec.get('action', 'N/A')[:80]}... — Priority: {badge}"):
                    st.write(f"**Action:** {rec.get('action', 'N/A')}")
                    st.write(f"**Evidence:** {rec.get('evidence', 'N/A')}")
                    st.write(f"**Expected Impact:** {rec.get('expected_impact', 'N/A')}")
                    st.write(f"**Risk Level:** {rec.get('risk_level', 'N/A')}")
        else:
            st.warning("Could not parse recommendations. Try again.")

st.divider()

# ── Section 7: CEO Briefing ───────────────────────────────────────────────────
st.header("Section 7: CEO Briefing")

if total_chunks == 0:
    st.info("No data yet.")
else:
    if st.button("Generate CEO Briefing", key="brief_btn", type="primary"):
        with st.spinner("Generating executive briefing..."):
            agent = load_agent()
            briefing = agent.generate_ceo_briefing()
            st.session_state["briefing"] = briefing

    if "briefing" in st.session_state:
        b = st.session_state["briefing"]
        st.subheader("What Happened?")
        st.write(b.get("what_happened", "N/A"))
        st.subheader("Why Does It Matter?")
        st.write(b.get("why_it_matters", "N/A"))
        st.subheader("What Should Management Do Next?")
        st.write(b.get("what_to_do", "N/A"))

    # free-form Q&A
    st.divider()
    st.subheader("Ask the CEO Agent")
    question = st.text_input("Your question:", placeholder=f"e.g. What should {TARGET_COMPANY} prioritize this quarter?")
    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            agent = load_agent()
            result = agent.query(question)
        st.markdown("**Answer:**")
        st.write(result["answer"])
        if result["sources"]:
            with st.expander("Sources used"):
                for s in result["sources"]:
                    st.write(f"[{s['source']}] {s['date']} — {s['text'][:150]}")
