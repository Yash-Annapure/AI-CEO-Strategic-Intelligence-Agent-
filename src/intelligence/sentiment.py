from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon", quiet=True)

# load these once at module level so we don't reload on every call
_news_pipeline = None
_vader = None


def _get_news_pipeline():
    global _news_pipeline
    if _news_pipeline is None:
        _news_pipeline = pipeline(
            "text-classification",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            truncation=True,
            max_length=512,
        )
    return _news_pipeline


def _get_vader():
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def analyze_news_sentiment(text):
    """takes a news article text string and runs it through a HuggingFace financial
    sentiment model. returns a dict with label (positive/negative/neutral) and score.

    how the score is calculated:
      distilroberta-finetuned-financial-news-sentiment-analysis is a transformer
      fine-tuned on financial news headlines. it outputs a softmax probability
      across 3 classes (positive, negative, neutral). sentiment_score is the
      winning class's probability (0.0 to 1.0) — i.e. how confident the model
      is in its classification."""
    pipe = _get_news_pipeline()
    # pipeline returns [{"label": "positive", "score": 0.94}] for the top class
    result = pipe(text[:512])[0]
    sentiment_label = result["label"].lower()
    confidence_score = round(result["score"], 4)
    return {
        "label": sentiment_label,
        "score": confidence_score,
    }


def analyze_reddit_sentiment(text):
    """takes a reddit post/comment string and runs NLTK VADER on it.
    VADER is rule-based and works well on short social media text.
    returns a dict with label and the compound score (-1 to +1).

    how the score is calculated:
      VADER assigns valence scores to individual words/phrases using a hand-crafted
      lexicon, then combines them with rules for emphasis, punctuation, and negation.
      polarity_scores() returns: pos, neg, neu (proportions of the text), and compound
      (normalized weighted sum, ranging from -1.0 most negative to +1.0 most positive).
      we use the compound score as sentiment_score and apply fixed thresholds to label:
        compound >= +0.05 → positive
        compound <= -0.05 → negative
        otherwise         → neutral"""
    vader = _get_vader()
    # raw_scores = {"neg": 0.12, "neu": 0.65, "pos": 0.23, "compound": 0.34}
    raw_scores = vader.polarity_scores(text)
    compound = raw_scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": round(compound, 4),
    }


def analyze_document(doc):
    """takes a cleaned document dict and picks the right sentiment analyzer
    based on the source field. reddit posts use VADER, everything else uses
    the HuggingFace financial news model. adds sentiment label and score to the doc."""
    source = doc.get("source", "").lower()
    text = doc.get("title", "") + " " + doc.get("text", "")
    text = text.strip()

    if not text:
        return {**doc, "sentiment": "neutral", "sentiment_score": 0.0}

    if "reddit" in source:
        result = analyze_reddit_sentiment(text)
    else:
        result = analyze_news_sentiment(text)

    return {
        **doc,
        "sentiment": result["label"],
        "sentiment_score": result["score"],
    }


def analyze_all(docs):
    """runs analyze_document on every doc in the list and returns them
    all with sentiment fields added."""
    results = []
    for i, doc in enumerate(docs):
        results.append(analyze_document(doc))
        if (i + 1) % 20 == 0:
            print(f"  Sentiment: {i + 1}/{len(docs)} done")
    print(f"Sentiment analysis complete: {len(results)} documents")
    return results


if __name__ == "__main__":
    news = {"title": "NVIDIA beats earnings expectations with record GPU sales", "text": "Revenue surged 122% year over year.", "source": "NewsAPI", "url": "https://example.com", "date": "2024-01-01"}
    reddit = {"title": "NVIDIA stock is way overvalued right now", "text": "The P/E ratio makes no sense, dumping my shares.", "source": "Reddit", "url": "https://reddit.com/1", "date": "2024-01-01"}

    print(analyze_document(news))
    print(analyze_document(reddit))
