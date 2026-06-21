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
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AI CEO AGENT (RAG)                        │
│                                                                 │
│   Query → Embed → Retrieve top-k chunks → Build prompt          │
│        → Qwen2.5-3B-Instruct → Strategic output                 │
│                                                                 │
│   Outputs: Opportunities / Risks / Recommendations / Briefing   │
└───────────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                          │
│                                                                 │
│   7 sections: Company Overview, Market Intelligence,            │
│   Opportunity Monitor, Risk Monitor, Sentiment Analysis,        │
│   Strategic Recommendations, CEO Briefing                       │
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
                   store.store()
                   - ChromaDB PersistentClient
                   - cosine similarity index
                   - skip if URL already exists
                        │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
            Dashboard        CEOAgent.query(question)
            reads metadata   - embed question
            directly from    - retrieve top-k chunks
            ChromaDB         - prompt + Qwen LLM
                             - return answer + sources
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
| LLM | `Qwen2.5-3B-Instruct` | open-source, instruction-tuned, fits on shared GPU |
| RAG framework | custom (no LangChain) | full control over retrieval and prompt logic |
| Dashboard | `Streamlit` | rapid prototyping, native Python |
| Package manager | `uv` | fast, reproducible environments |

---

## AI Pipeline

The agent uses **Retrieval-Augmented Generation (RAG)**:

1. **User submits a question** (or the system triggers a structured analysis)
2. The question is **embedded** using `all-MiniLM-L6-v2` into a 384-dim vector
3. ChromaDB performs **cosine similarity search** to retrieve the top-k most relevant chunks
4. Retrieved chunks are **formatted as context** with source and date labels
5. A **structured prompt** is built combining the context and the specific task (briefing / opportunities / risks / recommendations)
6. `Qwen2.5-3B-Instruct` **generates the response** grounded in the retrieved evidence
7. The response is **parsed** and displayed with source attribution

For structured outputs (opportunities, risks, recommendations), the prompts instruct the model to follow a strict labeled format, which is then parsed with regex into structured dicts for display.

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

**Why Qwen2.5-3B instead of 7B?**
The Datalab environment is a shared GPU server. The 7B model caused CUDA out-of-memory errors due to other users occupying GPU memory. The 3B model fits comfortably and produces coherent strategic outputs for this use case.

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
    │   └── vector_store.py        # ChromaDB interface
    ├── agent/
    │   └── ceo_agent.py           # RAG agent + structured analysis
    └── dashboard/
        └── app.py                 # Streamlit executive dashboard
```

---

## How to Run

```bash
# 1. Install dependencies
uv sync

# 2. Set up environment variables
cp .env.example .env  # fill in NEWS_API_KEY, MODEL_PATH

# 3. Run the full data pipeline
python main.py

# 4. Launch the dashboard
streamlit run src/dashboard/app.py --server.port 8508
```

On Datalab (no direct port access), expose via cloudflared:
```bash
./cloudflared tunnel --url http://localhost:8508
```
