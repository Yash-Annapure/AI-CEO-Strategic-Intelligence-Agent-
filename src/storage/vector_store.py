import chromadb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CHROMA_DB_PATH


class VectorStore:
    """this class wraps ChromaDB and provides three core operations:
    store embedded chunks, retrieve chunks by similarity, and check if a URL already exists."""

    def __init__(self, collection_name="nvidia_intel"):
        """sets up the ChromaDB client with a persistent local folder and
        gets or creates the collection we'll store everything in."""
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def url_exists(self, url):
        """checks if any chunk with this url is already stored so we avoid duplicates.
        returns True if the url is already in the collection, False otherwise."""
        results = self.collection.get(where={"url": url}, limit=1)
        return len(results["ids"]) > 0

    def store(self, chunks):
        """takes a list of embedded chunk dicts (output of embedder.embed_chunks)
        and adds them to ChromaDB. skips chunks whose url is already stored.
        each chunk needs: text, embedding, url, source, date, chunk_index."""
        added = 0
        skipped = 0

        for chunk in chunks:
            url = chunk.get("url", "")
            chunk_index = chunk.get("chunk_index", 0)
            doc_id = f"{url}_{chunk_index}"

            if self.url_exists(url) and chunk_index == 0:
                skipped += 1
                continue

            self.collection.add(
                ids=[doc_id],
                embeddings=[chunk["embedding"].tolist()],
                documents=[chunk["text"]],
                metadatas=[{
                    "url": url,
                    "source": chunk.get("source", ""),
                    "date": chunk.get("date", ""),
                    "chunk_index": chunk_index,
                }],
            )
            added += 1

        print(f"Stored: {added} chunks added, {skipped} documents skipped (already exist)")
        return added

    def retrieve(self, query_embedding, n_results=5):
        """takes a query embedding (numpy array) and returns the n most similar
        chunks from the collection. returns a list of dicts with text and metadata."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return chunks

    def count(self):
        """returns the total number of chunks stored in the collection."""
        return self.collection.count()


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    store = VectorStore()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sample_chunks = [
        {"text": "NVIDIA reported record revenue of $22B in Q3 2024", "url": "https://example.com/1", "source": "test", "date": "2024-01-01", "chunk_index": 0, "embedding": model.encode("NVIDIA reported record revenue of $22B in Q3 2024")},
        {"text": "Jensen Huang announced new Blackwell GPU architecture", "url": "https://example.com/2", "source": "test", "date": "2024-01-01", "chunk_index": 0, "embedding": model.encode("Jensen Huang announced new Blackwell GPU architecture")},
    ]

    store.store(sample_chunks)
    print(f"Total chunks in store: {store.count()}")

    query_embedding = model.encode("NVIDIA earnings revenue")
    results = store.retrieve(query_embedding, n_results=2)
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['text'][:80]}")
