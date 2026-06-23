import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import NEWS_API_KEY, TARGET_COMPANY

BASE_URL = "https://newsapi.org/v2/everything"


def fetch_news(query, page_size=10):
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    response = requests.get(BASE_URL, params=params, timeout=10)

    if response.status_code != 200:
        print(f"Failed to fetch news: {response.status_code} - {response.json().get('message')}")
        return []

    articles = response.json().get("articles", [])
    results = []

    for article in articles:
        title = article.get("title", "") or ""
        text = article.get("content", "") or ""

        if not title and not text:
            continue

        results.append({
            "title": title,
            "text": text,
            "url": article.get("url", ""),
            "source": article.get("source", {}).get("name", "NewsAPI"),
            "date": article.get("publishedAt", ""),
            "score": 0,
            "num_comments": 0,
        })

    return results


def collect():
    print(f"Fetching news for: {TARGET_COMPANY}")
    articles = fetch_news(TARGET_COMPANY, page_size=100)
    print(f"Collected {len(articles)} articles")
    return articles


if __name__ == "__main__":
    articles = collect()
    if articles:
        print(f"\nSample article:")
        print(f"  Title: {articles[0]['title']}")
        print(f"  Source: {articles[0]['source']}")
        print(f"  Date: {articles[0]['date']}")
