from sentence_transformers import SentenceTransformer

class Embedder:
    """this class loads the sentence transformer model once and provides methods
    to generate vector embeddings from text."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """loads the embedding model when the class is instantiated."""
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        """this func takes a single string and returns its vector embedding
        as a numpy array."""
        return self.model.encode(text)

    def embed_chunks(self, chunks):
        """this func takes the list of chunk dicts from the chunker,
        generates an embedding for each chunk's text field,
        and adds it back into the dict under the 'embedding' key.
        returns the same list of dicts with embeddings added."""

        for chunk in chunks:
            text = chunk["text"]  # get the "text" field from the chunk dict

            chunk["embedding"] = self.embed(text)  # call self.embed() on the text

        return chunks  # return the updated chunks list


if __name__ == "__main__":
    embedder = Embedder()
    sample_chunks = [
        {"text": "NVIDIA reported record revenue of $22B", "url": "https://example.com", "source": "test", "date": "2024-01-01", "chunk_index": 0},
        {"text": "Jensen Huang announced new Blackwell GPUs", "url": "https://example.com", "source": "test", "date": "2024-01-01", "chunk_index": 1},
    ]
    result = embedder.embed_chunks(sample_chunks)
    print(f"Chunks embedded: {len(result)}")
    print(f"Embedding size: {len(result[0]['embedding'])}")
