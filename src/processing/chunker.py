def chunk_text(text, chunk_size=500, overlap=50):
    """this func takes a long string of text and breaks it into smaller chunks of chunk_size
    characters, with an overlap between consecutive chunks to preserve context across boundaries.
    returns a list of text strings."""

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_document(doc, chunk_size=500, overlap=50):
    """this func takes a single cleaned document dict and splits it into chunks.
    it combines the title and text first so each chunk carries both signals,
    then attaches the original metadata to every chunk so we know where it came from.
    returns a list of chunk dicts, each with a chunk_index so we can track order."""

    title = doc.get("title", "")
    text = doc.get("text", "")
    full_text = f"{title}. {text}".strip()

    text_chunks = chunk_text(full_text, chunk_size, overlap)

    chunked_docs = []
    for i, chunk in enumerate(text_chunks):
        chunked_docs.append({
            "text": chunk,
            "url": doc.get("url", ""),
            "source": doc.get("source", ""),
            "date": doc.get("date", ""),
            "score": doc.get("score", 0),
            "num_comments": doc.get("num_comments", 0),
            "chunk_index": i,
            "sentiment": doc.get("sentiment", "neutral"),
            "sentiment_score": doc.get("sentiment_score", 0.0),
            "topic": doc.get("topic", "unknown"),
            "topic_score": doc.get("topic_score", 0.0),
        })

    return chunked_docs


def chunk_all(docs, chunk_size=500, overlap=50):
    """this func takes the full list of cleaned docs and runs chunk_document on each one,
    collecting all the resulting chunks into a single flat list ready for embedding."""

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)

    print(f"Chunked: {len(docs)} documents → {len(all_chunks)} chunks")
    return all_chunks
