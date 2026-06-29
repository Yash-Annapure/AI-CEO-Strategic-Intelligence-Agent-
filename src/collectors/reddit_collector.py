import feedparser
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import TARGET_COMPANY

SUBREDDITS = ["nvidia", "stocks", "investing", "technology", "artificial"]


_NO_CACHE_HEADERS = {"Cache-Control": "no-cache", "Pragma": "no-cache"}


def fetch_subreddit_posts(subreddit, limit=50):
    url = f"https://www.reddit.com/r/{subreddit}/search.rss?q={TARGET_COMPANY}&restrict_sr=1&sort=new&limit={limit}"
    feed = feedparser.parse(url, request_headers=_NO_CACHE_HEADERS)

    if feed.bozo and not feed.entries:
        print(f"Failed to fetch r/{subreddit}")
        return []

    results = []
    for entry in feed.entries:
        title = entry.get("title", "")
        text = entry.get("summary", "")

        results.append({
            "title": title,
            "text": text,
            "url": entry.get("link", ""),
            "source": f"reddit_r/{subreddit}",
            "date": entry.get("published", ""),
            "score": 0,
            "num_comments": 0,
        })

    return results


def collect(limit=50):
    all_posts = []
    for subreddit in SUBREDDITS:
        posts = fetch_subreddit_posts(subreddit, limit)
        print(f"r/{subreddit}: {len(posts)} posts mentioning {TARGET_COMPANY}")
        all_posts.extend(posts)
    return all_posts


if __name__ == "__main__":
    posts = collect()
    print(f"\nTotal collected: {len(posts)} posts")
    if posts:
        print(f"\nSample post:")
        print(f"  Title: {posts[0]['title']}")
        print(f"  Source: {posts[0]['source']}")
        print(f"  Date: {posts[0]['date']}")
