import feedparser
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import TARGET_COMPANY

SEARCH_TERMS = [TARGET_COMPANY.lower(), "jensen huang", "gpu", "ai chip"]

_NO_CACHE_HEADERS = {"Cache-Control": "no-cache", "Pragma": "no-cache"}


def _build_rss_feeds():
    after = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    return {
        "Google News NVIDIA": f"https://news.google.com/rss/search?q={TARGET_COMPANY}+stock+AI+after:{after}&hl=en-US&gl=US&ceid=US:en",
        "Google News Jensen": f"https://news.google.com/rss/search?q=Jensen+Huang+NVIDIA+after:{after}&hl=en-US&gl=US&ceid=US:en",
    }


def fetch_feed(name, url):
    feed = feedparser.parse(url, request_headers=_NO_CACHE_HEADERS)

    if feed.bozo and not feed.entries:
        print(f"Failed to fetch {name}")
        return []

    results = []
    for entry in feed.entries:
        title = entry.get("title", "")
        text = entry.get("summary", "") or entry.get("description", "")

        combined = (title + " " + text).lower()
        if not any(term in combined for term in SEARCH_TERMS):
            continue

        results.append({
            "title": title,
            "text": text,
            "url": entry.get("link", ""),
            "source": name,
            "date": entry.get("published", ""),
            "score": 0,
            "num_comments": 0,
        })

    return results


def collect():
    all_articles = []
    for name, url in _build_rss_feeds().items():
        articles = fetch_feed(name, url)
        print(f"{name}: {len(articles)} articles mentioning {TARGET_COMPANY}")
        all_articles.extend(articles)
    return all_articles


if __name__ == "__main__":
    articles = collect()
    print(f"\nTotal collected: {len(articles)} articles")
    if articles:
        print(f"\nSample article:")
        print(f"  Title: {articles[0]['title']}")
        print(f"  Source: {articles[0]['source']}")
        print(f"  Date: {articles[0]['date']}")
