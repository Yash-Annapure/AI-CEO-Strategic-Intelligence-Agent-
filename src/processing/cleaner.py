import html
import re
from bs4 import BeautifulSoup


def clean_html(raw_text):
    """this func takes in raw text that might have html tags and entities in it,
    it first decodes things like &amp; back to & then strips out all the html tags
    and returns just the plain readable text."""
    unescaped = html.unescape(raw_text)
    soup = BeautifulSoup(unescaped, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def clean_text(text):
    """this func takes plain text and removes the noise we dont want,
    it strips urls, any weird symbols that arent punctuation, and collapses
    multiple spaces or newlines into a single space."""
    
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s\.\!\?\,\:\;\'\"\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_document(doc):
    """this func takes a single document dict from any of our collectors,
    runs the title and text through html cleaning and text cleaning,
    and returns a cleaned version of the same dict. returns None if both
    title and text end up empty after cleaning."""
    
    title = clean_text(clean_html(doc.get("title", "")))
    text = clean_text(clean_html(doc.get("text", "")))

    if not title and not text:
        return None

    return {
        "title": title,
        "text": text,
        "url": doc.get("url", ""),
        "source": doc.get("source", ""),
        "date": doc.get("date", ""),
        "score": doc.get("score", 0),
        "num_comments": doc.get("num_comments", 0),
    }


def clean_all(docs):
    """this func takes the full list of docs from all collectors combined,
    deduplicates them by url so we dont store the same article twice,
    then cleans each one and returns the final list ready for chunking."""
    
    seen_urls = set()
    cleaned = []

    for doc in docs:
        url = doc.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)

        result = clean_document(doc)
        if result:
            cleaned.append(result)

    print(f"Cleaned: {len(cleaned)} documents (removed {len(docs) - len(cleaned)} duplicates/empties)")
    return cleaned


if __name__ == "__main__":
    sample = [
        {"title": "<b>NVIDIA</b> hits $22B revenue &amp; record margins!", "text": "<p>Jensen Huang announced...</p>", "url": "https://example.com/1", "source": "test", "date": "2024-01-01", "score": 0, "num_comments": 0},
        {"title": "<b>NVIDIA</b> hits $22B revenue &amp; record margins!", "text": "<p>Jensen Huang announced...</p>", "url": "https://example.com/1", "source": "test", "date": "2024-01-01", "score": 0, "num_comments": 0},
    ]
    result = clean_all(sample)
    print(f"Result: {len(result)} documents")
    print(f"Title: {result[0]['title']}")
