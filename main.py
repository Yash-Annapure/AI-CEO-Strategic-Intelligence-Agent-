"""
Full pipeline runner — use this on Datalab to collect, process, and store all data.
Run the dashboard separately with: streamlit run src/dashboard/app.py
"""
import sys
import os
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
from config import TARGET_COMPANY


def run_pipeline():
    print(f"\n=== AI CEO Strategic Intelligence Agent — {TARGET_COMPANY} ===\n")

    # Step 1: Collect
    print("Step 1/6: Collecting data...")
    raw = collect_reddit() + collect_news() + collect_rss()
    print(f"  Total raw documents: {len(raw)}")

    # Step 2: Clean + deduplicate
    print("\nStep 2/6: Cleaning and deduplicating...")
    cleaned = clean_all(raw)

    # Step 3: Sentiment
    print("\nStep 3/6: Sentiment analysis...")
    cleaned = analyze_all(cleaned)

    # Step 4: Topic classification
    print("\nStep 4/6: Topic classification...")
    cleaned = classify_all(cleaned)

    # Step 5: Chunk
    print("\nStep 5/6: Chunking...")
    chunks = chunk_all(cleaned)

    # Step 6: Embed + store
    print("\nStep 6/6: Embedding and storing...")
    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)
    store = VectorStore()
    store.clear()
    added = store.store(embedded)

    print(f"\n=== Pipeline complete ===")
    print(f"  Documents collected:  {len(raw)}")
    print(f"  Documents cleaned:    {len(cleaned)}")
    print(f"  Chunks stored:        {added}")
    print(f"  Total in store:       {store.count()}")
    print(f"\nTo launch dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    run_pipeline()
