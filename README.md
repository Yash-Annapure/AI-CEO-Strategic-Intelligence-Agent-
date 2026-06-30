# AI CEO: Strategic Intelligence Agent

An AI-powered system that continuously collects live information about NVIDIA from multiple public sources, processes and stores it in a vector database, and uses a local open-source LLM to generate executive-level strategic recommendations — answering the question: *"If you were the CEO today, what would you do next and why?"*

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│                                                                 │
│   Reddit (RSS)        NewsAPI          Google News (RSS)        │
│   r/nvidia            Financial news   NVIDIA keyword           │
│   r/stocks            100 articles     Jensen keyword           │
│   r/investing                          ~190 articles            │
└──────────────┬──────────────┬──────────────────┬────────────────┘
               │              │                  │
               └──────────────┴──────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING PIPELINE                         │
│                                                                 │
│   1. Cleaner    → strip HTML, remove noise, deduplicate by URL  │
│   2. Sentiment  → HuggingFace (news) + NLTK VADER (Reddit)      │
│   3. Classifier → zero-shot topic classification (bart-mnli)    │
│   4. Chunker    → split into 500-char chunks with 50 overlap    │
│   5. Embedder   → all-MiniLM-L6-v2 → 384-dim vectors            │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE REPOSITORY                        │
│                                                                 │
│   ChromaDB (persistent, cosine similarity)                      │
│   Each chunk stored with: text, embedding, source, date,        │
│   sentiment, sentiment_score, topic, topic_score, url           │
│   Cleared and rebuilt on every pipeline run (no stale data)     │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH REACT AGENT                          │
│                                                                 │
│   Goal → LLM decides tool → tool executes → observe → loop     │
│                                                                 │
│   Tools:  search_knowledge · detect_opportunities · assess_risks │
│           generate_recommendations · get_ceo_briefing_context   │
│           get_sentiment_summary · get_topic_summary             │
│                                                                 │
│   LLM: Qwen2.5-3b via Ollama (ChatOllama)                       │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                          │
│                                                                 │
│   7 sections: Company Overview · Market Intelligence            │
│               Opportunity Monitor · Risk Monitor                │
│               Sentiment Analysis · Strategic Recommendations    │
│               CEO Briefing (pre-computed + live agent)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Reddit RSS ──┐
NewsAPI ─────┼──► collect() ──► raw docs list
Google RSS ──┘         │
                        │
                        ▼
                   clean_all()
                   - decode HTML
                   - remove URLs/symbols
                   - deduplicate by URL
                        │
                        ▼
                   analyze_all()
                   - news → distilroberta financial sentiment
                   - reddit → NLTK VADER compound score
                        │
                        ▼
                   classify_all()
                   - facebook/bart-large-mnli
                   - 8 topic labels (zero-shot)
                        │
                        ▼
                   chunk_all()
                   - 500 chars, 50 overlap
                   - title prepended to each chunk
                        │
                        ▼
                   embed_chunks()
                   - all-MiniLM-L6-v2
                   - 384-dimensional vectors
                        │
                        ▼
                   store.clear() → store.store()
                   - ChromaDB PersistentClient
                   - cosine similarity index
                   - purged and rebuilt every run
                        │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
            Dashboard        LangGraph ReAct Agent
            Section 1+2:     - LLM chooses tools
            reads metadata   - search_knowledge →
            directly from      RAGRetriever embeds
            ChromaDB           query, cosine search
                             - LLM reads chunks,
                               generates answer
```

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Data collection | `requests`, `feedparser` | lightweight, no auth needed for RSS |
| HTML cleaning | `BeautifulSoup4` | handles malformed HTML from news APIs |
| News sentiment | `distilroberta-finetuned-financial-news-sentiment-analysis` | fine-tuned on financial text |
| Reddit sentiment | `NLTK VADER` | rule-based, works well on short social text |
| Topic classification | `facebook/bart-large-mnli` | zero-shot, no labelled training data needed |
| Embeddings | `all-MiniLM-L6-v2` | fast, 384-dim, strong semantic similarity |
| Vector store | `ChromaDB` | persistent, easy cosine similarity retrieval |
| LLM | `Qwen2.5-3b` via Ollama | served locally via OpenAI-compatible API, tool-calling support |
| Agent framework | `LangGraph` `create_react_agent` | handles ReAct loop automatically; tools bound to ChatOllama |
| RAG retriever | `RAGRetriever` (retriever.py) | dedicated retrieval-only component: embed → cosine search → chunks |
| Dashboard | `Streamlit` | rapid prototyping, native Python |
| Package manager | `uv` | fast, reproducible environments |

---

## AI Pipeline

The agent uses a **LangGraph ReAct loop** (Reason + Act) backed by **RAG**:

1. **User submits a strategic goal** via the dashboard
2. `create_react_agent` starts the loop — the LLM reads the goal and decides which tool to call
3. **Tool executes** — e.g. `search_knowledge(query)` calls `RAGRetriever`:
   - embeds the query with `all-MiniLM-L6-v2` → 384-dim vector
   - cosine similarity search in ChromaDB → top-k chunks returned as text
4. **LLM observes** the tool result (the retrieved chunks) and decides: call another tool or produce a final answer
5. Loop repeats until the LLM produces an answer with no tool call
6. Dashboard shows the **tool trace** (which tools were called and with what args) + the **final answer**

The retrieval (R) and generation (G) halves of RAG are in separate files: `retriever.py` handles embed → search → return chunks; `langgraph_agent.py` handles the LLM loop that generates the final grounded response.

---

## Design Decisions

**Why RSS instead of Reddit API?**
Reddit's official API requires OAuth and has strict rate limits. RSS feeds for subreddits are publicly available and return the same content without authentication.

**Why two sentiment models?**
News articles and Reddit posts have very different language styles. Financial news is formal and measured — a fine-tuned transformer model captures its nuance better. Reddit is short, informal, and emoji-heavy — VADER's rule-based approach handles this better and runs without GPU.

**Why zero-shot classification for topics?**
We don't have labelled training data for NVIDIA-specific topics. Zero-shot classification with `bart-large-mnli` lets us define custom topic labels and classify without any fine-tuning.

**Why ChromaDB over FAISS?**
ChromaDB is persistent by default (survives restarts), supports metadata filtering, and has a simple Python API. FAISS is faster but requires manual index serialization and doesn't natively store metadata alongside vectors.

**Why chunk at 500 characters with 50 overlap?**
The embedding model has a 256-token limit. 500 characters is approximately 100-150 tokens — safely within limits. The 50-character overlap ensures that sentences split across chunk boundaries are still captured in at least one chunk.

**Why LangGraph instead of a custom RAG loop?**
LangGraph's `create_react_agent` handles the ReAct loop automatically — the LLM decides which tool to call, the tool executes, the result is fed back, and the loop continues until the LLM produces a final answer. This makes the agent behavior explicit and explainable: every decision is a tool call the professor can see in the tool trace.

**Why Ollama instead of loading Qwen via HuggingFace?**
Ollama serves the model via a local OpenAI-compatible API, which means `ChatOllama` can bind tools natively. Qwen2.5 supports tool calling in this mode, making the ReAct loop reliable without complex prompt engineering.

**Why clear ChromaDB on every run?**
The original deduplication-by-URL approach caused stale data to persist across runs — old articles were never refreshed. Clearing the collection before each pipeline run ensures the knowledge base always reflects the latest collected data.

---

## Project Structure

```
AI-CEO-Strategic-Intelligence-Agent/
├── main.py                        # headless pipeline runner
├── config.py                      # loads .env variables
├── pyproject.toml                 # dependencies (uv)
└── src/
    ├── collectors/
    │   ├── reddit_collector.py    # RSS-based Reddit scraper
    │   ├── news_collector.py      # NewsAPI client
    │   └── rss_collector.py       # Google News RSS parser
    ├── processing/
    │   ├── cleaner.py             # HTML cleaning, deduplication
    │   ├── chunker.py             # text splitting with overlap
    │   └── embedder.py            # sentence-transformers wrapper
    ├── intelligence/
    │   ├── sentiment.py           # dual-model sentiment analysis
    │   └── classifier.py         # zero-shot topic classification
    ├── storage/
    │   └── vector_store.py        # ChromaDB interface + clear()
    ├── rag/
    │   └── retriever.py           # RAGRetriever: embed → cosine search → chunks
    ├── agent/
    │   └── langgraph_agent.py     # LangGraph ReAct agent + 7 tools
    └── dashboard/
        └── app.py                 # Streamlit dashboard (7 sections)
```

---

## How to Run

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment variables
cp .env.example .env  # fill in NEWS_API_KEY, MODEL_PATH

# 3. Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve                  # keep this terminal open

# 4. Pull the model (new terminal)
ollama pull qwen2.5:3b

# 5. Launch the dashboard
uv run streamlit run src/dashboard/app.py

# 6. Then click "Run Full Pipeline" from the dashboard sidebar
#    (collects data, runs NLP pipeline, and runs the agent — no need to run main.py separately)
```

On Datalab (no direct port access), expose via Cloudflare tunnel:
```bash
# Install cloudflared (one-time)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared && chmod +x cloudflared

# Run the tunnel
./cloudflared tunnel --url http://localhost:8501
```

If `apt-get install zstd` fails during Ollama install:
```bash
sudo apt-get update && sudo apt-get install -y zstd
```
