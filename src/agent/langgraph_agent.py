import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from src.rag.retriever import RAGRetriever
from src.storage.vector_store import VectorStore
from config import TARGET_COMPANY

# module-level singleton so the embedding model loads only once per session
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(n_results=5)
    return _retriever


# ── Tool definitions ──────────────────────────────────────────────────────────
# each @tool function is a capability the agent can autonomously choose to call.
# the docstring is what the LLM reads to decide when to use each tool.

@tool
def search_knowledge(query: str) -> str:
    """Search the NVIDIA knowledge base for relevant news and intelligence.
    Use this to retrieve evidence before making any strategic claim."""
    return _get_retriever().retrieve_as_text(query)


@tool
def detect_opportunities() -> str:
    """Retrieve intelligence about strategic opportunities for NVIDIA:
    emerging technologies, new markets, partnerships, product opportunities.
    Call this tool before identifying or listing opportunities."""
    return _get_retriever().retrieve_as_text(
        f"NVIDIA strategic opportunities emerging technology new markets partnerships growth expansion"
    )


@tool
def assess_risks() -> str:
    """Retrieve intelligence about risks and threats facing NVIDIA:
    competitive threats, regulatory changes, negative sentiment, supply chain issues.
    Call this tool before identifying or listing risks."""
    return _get_retriever().retrieve_as_text(
        f"NVIDIA risks threats competition regulatory legal supply chain challenges negative"
    )


@tool
def generate_recommendations() -> str:
    """Retrieve strategic context needed to generate CEO-level recommendations for NVIDIA.
    Call this tool before producing strategic recommendations."""
    return _get_retriever().retrieve_as_text(
        f"NVIDIA strategic direction CEO priorities competitive positioning market leadership decisions"
    )


@tool
def get_ceo_briefing_context() -> str:
    """Retrieve broad market context needed to write an executive CEO briefing for NVIDIA.
    Call this tool before generating the CEO briefing."""
    return _get_retriever().retrieve_as_text(
        f"NVIDIA recent developments market position news announcements industry trends"
    )


@tool
def get_sentiment_summary() -> str:
    """Get the overall market sentiment breakdown across all collected documents.
    Returns counts of positive, neutral, and negative articles."""
    store = VectorStore()
    meta = store.collection.get(include=["metadatas"])["metadatas"]
    if not meta:
        return "No data in knowledge base."
    counts = {}
    for m in meta:
        label = m.get("sentiment", "unknown")
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    lines = [
        f"{label}: {count} ({round(count / total * 100)}%)"
        for label, count in sorted(counts.items())
    ]
    return "Sentiment breakdown:\n" + "\n".join(lines)


@tool
def get_topic_summary() -> str:
    """Get the topic distribution across all collected documents.
    Use this to understand which themes dominate the current intelligence."""
    store = VectorStore()
    meta = store.collection.get(include=["metadatas"])["metadatas"]
    if not meta:
        return "No data in knowledge base."
    counts = {}
    for m in meta:
        topic = m.get("topic", "unknown")
        counts[topic] = counts.get(topic, 0) + 1
    total = sum(counts.values())
    lines = [
        f"{topic}: {count} ({round(count / total * 100)}%)"
        for topic, count in sorted(counts.items(), key=lambda x: -x[1])
    ]
    return "Topic breakdown:\n" + "\n".join(lines)


# ── Agent runner ──────────────────────────────────────────────────────────────

def run_agent(goal: str, ollama_model: str = "qwen2.5:3b") -> dict:
    """runs the LangGraph ReAct agent on a strategic goal.

    the agent loop (handled automatically by create_react_agent):
      1. LLM reads the goal and decides which tool to call
      2. tool runs, result fed back to LLM
      3. LLM decides: call another tool or give final answer
      4. loop ends when LLM produces an answer with no tool call

    returns: {"answer": str, "tool_calls": list of {"tool": str, "input": dict}}
    """
    llm = ChatOllama(model=ollama_model, num_predict=1024)
    tools = [
        search_knowledge,
        detect_opportunities,
        assess_risks,
        generate_recommendations,
        get_ceo_briefing_context,
        get_sentiment_summary,
        get_topic_summary,
    ]

    system_prompt = (
        f"You are a strategic intelligence analyst for {TARGET_COMPANY}. "
        "Rules: "
        "1. Always call at least one tool before drawing conclusions. "
        "2. Be specific — name the sources you found (e.g. Barron's, Reddit). "
        "3. Give 3-5 concrete bullet points, no vague generalities. "
        "4. End with a clear, definitive recommendation. "
        "5. Always complete your response — never trail off mid-sentence."
    )

    agent = create_react_agent(llm, tools, prompt=system_prompt)
    result = agent.invoke({"messages": [("user", goal)]})

    # parse the message trace to extract tool calls and final answer
    tool_calls_made = []
    final_answer = ""

    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append({
                    "tool": tc["name"],
                    "input": tc["args"],
                })

    # the final answer is the last AIMessage that has content but no tool calls
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_answer = msg.content
            break

    return {
        "answer": final_answer,
        "tool_calls": tool_calls_made,
    }
