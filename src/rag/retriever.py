import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.processing.embedder import Embedder
from src.storage.vector_store import VectorStore


class RAGRetriever:
    """the RAG component. embeds a query, searches ChromaDB by cosine similarity,
    and returns the most relevant chunks from the knowledge base.

    this is the retrieval half of Retrieval-Augmented Generation — the LLM is
    never called here. this class only handles: embed → search → return chunks."""

    def __init__(self, n_results=5):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.n_results = n_results

    def retrieve(self, query: str) -> list:
        """embeds the query string and retrieves the top n most similar chunks
        from ChromaDB. returns a list of dicts with text, metadata, and distance."""
        query_embedding = self.embedder.embed(query)
        return self.store.retrieve(query_embedding, n_results=self.n_results)

    def retrieve_as_text(self, query: str) -> str:
        """retrieves chunks and formats them as a numbered text block ready to be
        injected into an LLM prompt as grounding context."""
        chunks = self.retrieve(query)
        if not chunks:
            return "No relevant information found in the knowledge base."
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "unknown")
            date = chunk["metadata"].get("date", "")[:10]
            parts.append(f"[{i+1}] ({source}, {date})\n{chunk['text']}")
        return "\n\n".join(parts)
