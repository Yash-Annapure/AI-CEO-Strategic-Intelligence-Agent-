import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import MODEL_PATH, TARGET_COMPANY
from src.storage.vector_store import VectorStore
from src.processing.embedder import Embedder


class CEOAgent:
    """this is the core RAG agent. it takes a question, retrieves the most relevant
    chunks from ChromaDB using semantic search, and passes them as context to the
    local Qwen LLM to generate a CEO-level strategic insight."""

    def __init__(self, model_path=None, n_results=5):
        self.n_results = n_results
        self.embedder = Embedder()
        self.store = VectorStore()

        path = model_path or MODEL_PATH
        print(f"Loading LLM from: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("LLM loaded.")

    def _build_prompt(self, question, context_chunks):
        """builds the RAG prompt by injecting retrieved context chunks before
        the question so the LLM can ground its answer in real data."""
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk["metadata"].get("source", "unknown")
            date = chunk["metadata"].get("date", "")[:10]
            context_parts.append(f"[{i+1}] ({source}, {date})\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a strategic intelligence assistant helping a CEO make decisions about {TARGET_COMPANY}.

Use the following news and market intelligence to answer the question. Be concise, factual, and strategic.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        return prompt

    def query(self, question):
        """takes a natural language question, embeds it, retrieves the top n_results
        chunks from ChromaDB, builds a grounded prompt, and returns the LLM's answer
        along with the source chunks used."""
        query_embedding = self.embedder.embed(question)
        chunks = self.store.retrieve(query_embedding, n_results=self.n_results)

        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base. Please run the data pipeline first.",
                "sources": [],
            }

        prompt = self._build_prompt(question, chunks)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded[len(prompt):].strip()

        return {
            "answer": answer,
            "sources": [
                {
                    "text": c["text"][:200],
                    "source": c["metadata"].get("source", ""),
                    "date": c["metadata"].get("date", "")[:10],
                    "url": c["metadata"].get("url", ""),
                }
                for c in chunks
            ],
        }


if __name__ == "__main__":
    agent = CEOAgent()
    result = agent.query(f"What are the biggest risks facing {TARGET_COMPANY} right now?")
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"  [{s['source']}] {s['text'][:100]}")
