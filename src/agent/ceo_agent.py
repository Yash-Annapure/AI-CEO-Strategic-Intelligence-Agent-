import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import MODEL_PATH, TARGET_COMPANY
from src.storage.vector_store import VectorStore
from src.processing.embedder import Embedder


class CEOAgent:
    """the core RAG agent. retrieves relevant chunks from ChromaDB and uses
    the local Qwen LLM to generate strategic intelligence outputs."""

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

    def _retrieve(self, query_text, n_results=None):
        """embeds the query text and retrieves the top n most similar chunks."""
        embedding = self.embedder.embed(query_text)
        return self.store.retrieve(embedding, n_results=n_results or self.n_results)

    def _generate(self, prompt, max_new_tokens=512):
        """runs the LLM on the given prompt and returns the generated text only."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    def _format_context(self, chunks):
        """formats retrieved chunks into a numbered context block for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "unknown")
            date = chunk["metadata"].get("date", "")[:10]
            parts.append(f"[{i+1}] ({source}, {date})\n{chunk['text']}")
        return "\n\n".join(parts)

    def _parse_items(self, raw_text, item_prefix, fields):
        """parses the structured LLM output into a list of dicts."""
        items = []
        pattern = rf"{item_prefix}\s+\d+:"
        parts = re.split(pattern, raw_text, flags=re.IGNORECASE)
        blocks = [p.strip() for p in parts if p.strip()]

        for block in blocks:
            item = {}
            for field in fields:
                match = re.search(rf"{re.escape(field)}:\s*(.+?)(?=\n[A-Z]|\Z)", block, re.IGNORECASE | re.DOTALL)
                item[field.lower().replace(" ", "_")] = match.group(1).strip() if match else "N/A"
            if any(v != "N/A" for v in item.values()):
                items.append(item)

        return items[:3]

    def _parse_briefing(self, raw_text):
        """parses the CEO briefing sections from LLM output."""
        sections = {}
        keys = {
            "what_happened": r"WHAT HAPPENED:\s*(.+?)(?=WHY IT MATTERS:|WHAT MANAGEMENT|\Z)",
            "why_it_matters": r"WHY IT MATTERS:\s*(.+?)(?=WHAT MANAGEMENT|\Z)",
            "what_to_do": r"WHAT MANAGEMENT SHOULD DO NEXT:\s*(.+?)(?=\Z)",
        }
        for key, pattern in keys.items():
            match = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
            text = match.group(1).strip() if match else raw_text[:300]
            # strip any prompt template placeholders the model echoed back
            text = re.sub(r'\[.*?\]', '', text).strip()
            sections[key] = text
        return sections

    def query(self, question):
        """answers a free-form strategic question using RAG."""
        chunks = self._retrieve(question)
        if not chunks:
            return {"answer": "No relevant information found. Run the data pipeline first.", "sources": []}

        context = self._format_context(chunks)
        prompt = f"""You are a strategic intelligence assistant advising the CEO of {TARGET_COMPANY}.

Use the following intelligence to answer the question. Be concise, factual, and strategic.

INTELLIGENCE:
{context}

QUESTION: {question}

ANSWER:"""

        answer = self._generate(prompt)
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

    def analyze_opportunities(self):
        """retrieves relevant chunks and identifies 3 strategic opportunities with
        title, impact level, evidence, and confidence score."""
        chunks = self._retrieve(f"{TARGET_COMPANY} opportunities growth markets technology partnerships", n_results=8)
        context = self._format_context(chunks)

        prompt = f"""You are a strategic analyst for {TARGET_COMPANY}. Based on the intelligence below, identify 3 key strategic opportunities.

INTELLIGENCE:
{context}

List exactly 3 opportunities using this exact format:

OPPORTUNITY 1:
Title: [short title]
Impact: [High/Medium/Low]
Evidence: [one sentence]
Confidence: [High/Medium/Low]

OPPORTUNITY 2:
Title: [short title]
Impact: [High/Medium/Low]
Evidence: [one sentence]
Confidence: [High/Medium/Low]

OPPORTUNITY 3:
Title: [short title]
Impact: [High/Medium/Low]
Evidence: [one sentence]
Confidence: [High/Medium/Low]

OPPORTUNITIES:"""

        raw = self._generate(prompt, max_new_tokens=400)
        return self._parse_items(raw, "OPPORTUNITY", ["Title", "Impact", "Evidence", "Confidence"])

    def analyze_risks(self):
        """retrieves relevant chunks and identifies 3 key risks with title,
        category, severity, evidence, and confidence score."""
        chunks = self._retrieve(f"{TARGET_COMPANY} risks threats competition regulation supply chain", n_results=8)
        context = self._format_context(chunks)

        prompt = f"""You are a risk analyst for {TARGET_COMPANY}. Based on the intelligence below, identify 3 key risks.

INTELLIGENCE:
{context}

List exactly 3 risks using this exact format:

RISK 1:
Title: [short title]
Category: [Competitive/Regulatory/Financial/Operational/Reputational]
Severity: [High/Medium/Low]
Evidence: [one sentence]
Confidence: [High/Medium/Low]

RISK 2:
Title: [short title]
Category: [Competitive/Regulatory/Financial/Operational/Reputational]
Severity: [High/Medium/Low]
Evidence: [one sentence]
Confidence: [High/Medium/Low]

RISK 3:
Title: [short title]
Category: [Competitive/Regulatory/Financial/Operational/Reputational]
Severity: [High/Medium/Low]
Evidence: [one sentence]
Confidence: [High/Medium/Low]

RISKS:"""

        raw = self._generate(prompt, max_new_tokens=400)
        return self._parse_items(raw, "RISK", ["Title", "Category", "Severity", "Evidence", "Confidence"])

    def generate_recommendations(self):
        """generates 3 structured strategic recommendations with action, priority,
        evidence, expected impact, and risk level."""
        chunks = self._retrieve(f"{TARGET_COMPANY} strategy investment priorities actions", n_results=8)
        context = self._format_context(chunks)

        prompt = f"""You are a strategic advisor to the CEO of {TARGET_COMPANY}. Based on the intelligence, provide 3 actionable strategic recommendations.

INTELLIGENCE:
{context}

List exactly 3 recommendations using this exact format:

RECOMMENDATION 1:
Action: [what the CEO should do]
Priority: [High/Medium/Low]
Evidence: [supporting evidence]
Expected Impact: [what outcome this drives]
Risk Level: [High/Medium/Low]

RECOMMENDATION 2:
Action: [what the CEO should do]
Priority: [High/Medium/Low]
Evidence: [supporting evidence]
Expected Impact: [what outcome this drives]
Risk Level: [High/Medium/Low]

RECOMMENDATION 3:
Action: [what the CEO should do]
Priority: [High/Medium/Low]
Evidence: [supporting evidence]
Expected Impact: [what outcome this drives]
Risk Level: [High/Medium/Low]

RECOMMENDATIONS:"""

        raw = self._generate(prompt, max_new_tokens=500)
        return self._parse_items(raw, "RECOMMENDATION", ["Action", "Priority", "Evidence", "Expected Impact", "Risk Level"])

    def generate_ceo_briefing(self):
        """generates a structured executive summary: what happened, why it matters,
        and what management should do next."""
        chunks = self._retrieve(f"{TARGET_COMPANY} latest news developments strategy", n_results=10)
        context = self._format_context(chunks)

        prompt = f"""You are preparing a CEO briefing for {TARGET_COMPANY} based on the latest market intelligence.

INTELLIGENCE:
{context}

Write a concise executive briefing using exactly this format:

WHAT HAPPENED:
[2-3 sentences on the most important recent developments]

WHY IT MATTERS:
[2-3 sentences on the business implications]

WHAT MANAGEMENT SHOULD DO NEXT:
[2-3 sentences with concrete next steps]

BRIEFING:"""

        raw = self._generate(prompt, max_new_tokens=400)
        return self._parse_briefing(raw)


if __name__ == "__main__":
    agent = CEOAgent()
    print("\n=== CEO BRIEFING ===")
    briefing = agent.generate_ceo_briefing()
    for k, v in briefing.items():
        print(f"\n{k.upper()}:\n{v}")
