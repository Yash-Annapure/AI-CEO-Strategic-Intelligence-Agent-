"""
Full pipeline runner — use this on Datalab to collect, process, and store all data.
Run the dashboard separately with: streamlit run src/dashboard/app.py
"""
import sys
import os
import json
import datetime
sys.path.append(os.path.dirname(__file__))

from src.collectors.reddit_collector import collect as collect_reddit
from src.collectors.news_collector import collect as collect_news
from src.collectors.rss_collector import collect as collect_rss
from src.processing.cleaner import clean_all
from src.processing.chunker import chunk_all
from src.processing.embedder import Embedder
from src.intelligence.sentiment import analyze_all
from src.intelligence.classifier import classify_all
from src.storage.vector_store import VectorStore
from config import TARGET_COMPANY, OLLAMA_MODEL

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "dashboard_data.json")

AGENT_GOALS = {
    "opportunities": (
        f"What are the top 3 strategic opportunities for {TARGET_COMPANY} "
        "based on the latest intelligence? Be specific and cite your sources."
    ),
    "risks": (
        f"What are the biggest risks facing {TARGET_COMPANY} right now? "
        "Be specific and cite your sources."
    ),
    "recommendations": (
        f"What are the top 3 strategic recommendations you would give "
        f"{TARGET_COMPANY}'s CEO based on the current intelligence? Be specific."
    ),
}


def run_pipeline():
    print(f"\n=== AI CEO Strategic Intelligence Agent — {TARGET_COMPANY} ===\n")

    # Step 1: Collect
    print("Step 1/7: Collecting data...")
    raw = collect_reddit() + collect_news() + collect_rss()
    print(f"  Total raw documents: {len(raw)}")

    # Step 2: Clean + deduplicate
    print("\nStep 2/7: Cleaning and deduplicating...")
    cleaned = clean_all(raw)

    # Step 3: Sentiment
    print("\nStep 3/7: Sentiment analysis...")
    cleaned = analyze_all(cleaned)

    # Step 4: Topic classification
    print("\nStep 4/7: Topic classification...")
    cleaned = classify_all(cleaned)

    # Step 5: Chunk
    print("\nStep 5/7: Chunking...")
    chunks = chunk_all(cleaned)

    # Step 6: Embed + store
    print("\nStep 6/7: Embedding and storing...")
    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)
    store = VectorStore()
    store.clear()
    added = store.store(embedded)

    # Step 7: Run agent for each dashboard section and save to JSON
    print("\nStep 7/7: Running strategic agent analysis...")
    try:
        from src.agent.langgraph_agent import run_agent
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        dashboard_data = {"generated_at": datetime.datetime.now().isoformat()}
        for key, goal in AGENT_GOALS.items():
            print(f"  Agent: {key}...")
            dashboard_data[key] = run_agent(goal, ollama_model=OLLAMA_MODEL)
        with open(RESULTS_PATH, "w") as f:
            json.dump(dashboard_data, f, indent=2)
        print(f"  Saved to results/dashboard_data.json")
    except Exception as e:
        print(f"  Agent step failed (is Ollama running?): {e}")
        print("  Dashboard sections 4-6 will be empty until you re-run the pipeline.")

    print(f"\n=== Pipeline complete ===")
    print(f"  Documents collected:  {len(raw)}")
    print(f"  Documents cleaned:    {len(cleaned)}")
    print(f"  Chunks stored:        {added}")
    print(f"  Total in store:       {store.count()}")
    print(f"\nTo launch dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    run_pipeline()
