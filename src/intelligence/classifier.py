from transformers import pipeline

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


TOPIC_LABELS = [
    "earnings and revenue",
    "product launch",
    "AI and machine learning",
    "competition and market share",
    "regulation and legal",
    "partnerships and acquisitions",
    "stock and investor sentiment",
    "supply chain",
]


def classify_document(doc):
    """takes a cleaned document dict and runs zero-shot classification to assign
    the most relevant topic from TOPIC_LABELS. adds topic and topic_score to the doc.

    how the score is calculated:
      bart-large-mnli computes Natural Language Inference (NLI) entailment between
      the article text and each candidate label phrase. the model asks: does this text
      ENTAIL (support) the label? the entailment logits are passed through softmax
      across all 8 labels, producing probabilities that sum to 1.0.
      topic_score is the winning label's probability — how confidently the model
      says this article belongs to that topic vs all other options."""
    pipe = _get_classifier()
    text = doc.get("title", "") + " " + doc.get("text", "")
    text = text.strip()[:512]

    if not text:
        return {**doc, "topic": "unknown", "topic_score": 0.0}

    # multi_label=False → softmax across labels (mutually exclusive classification)
    # result["labels"] is sorted by score descending
    # result["scores"] are the softmax probabilities for each label
    result = pipe(text, candidate_labels=TOPIC_LABELS, multi_label=False)

    winning_label = result["labels"][0]
    winning_probability = result["scores"][0]

    return {
        **doc,
        "topic": winning_label,
        "topic_score": round(winning_probability, 4),
    }


def classify_all(docs):
    """runs classify_document on every doc in the list and returns them
    all with topic fields added."""
    results = []
    for i, doc in enumerate(docs):
        results.append(classify_document(doc))
        if (i + 1) % 10 == 0:
            print(f"  Classification: {i + 1}/{len(docs)} done")
    print(f"Classification complete: {len(results)} documents")
    return results


if __name__ == "__main__":
    doc = {
        "title": "NVIDIA launches Blackwell B200 GPU for data centers",
        "text": "Jensen Huang unveiled the next generation GPU architecture at GTC 2024.",
        "source": "NewsAPI",
        "url": "https://example.com",
        "date": "2024-01-01",
    }
    result = classify_document(doc)
    print(f"Topic: {result['topic']} ({result['topic_score']})")
