# LangGraph Agent — Deep Dive

---

## The Big Picture

The agent answers strategic questions about NVIDIA. But instead of just sending a question directly to an LLM and hoping it knows the answer, the agent **retrieves real evidence first** and **shows its work**.

The flow:

```
User goal → LLM decides what tool to call → tool runs → LLM reads result → repeat or answer
```

This is called a **ReAct loop**.

---

## What is ReAct?

ReAct = **Re**ason + **Act**

The LLM alternates between two things:
- **Reason** — think about what information it needs
- **Act** — call a tool to get that information

It keeps looping until it has enough evidence to give a final answer.

Without ReAct, you'd just ask the LLM a question and it would answer from its training data — which may be outdated, wrong, or hallucinated. ReAct forces the model to **go find real evidence** before concluding.

---

## What is LangGraph?

LangGraph is a framework that manages the ReAct loop automatically. Specifically `create_react_agent` builds the loop for you:

```python
agent = create_react_agent(llm, tools, prompt=system_prompt)
result = agent.invoke({"messages": [("user", goal)]})
```

Without LangGraph you would have to manually:
- Send the goal to the LLM
- Check if it wants to call a tool
- Run the tool
- Feed the result back
- Check again if it wants another tool
- Keep looping until it stops

`create_react_agent` handles all of that. You just pass it the LLM, the tools, and the goal.

---

## ChatOllama

```python
llm = ChatOllama(model=ollama_model, num_predict=1024)
```

Ollama runs the LLM locally on the server. `ChatOllama` is the LangChain wrapper that lets us talk to it via an OpenAI-compatible API.

`num_predict=1024` sets the maximum number of tokens the model can generate in one response. Without this it defaults to something low and the response gets cut off mid-sentence.

**Why Ollama instead of loading the model directly with HuggingFace?**
Ollama serves the model via a local API. This means `ChatOllama` can bind tools natively — Qwen2.5 supports tool calling in this mode, which is what makes the ReAct loop work reliably.

---

## The @tool Decorator

```python
@tool
def search_knowledge(query: str) -> str:
    """Search the NVIDIA knowledge base for relevant news and intelligence.
    Use this to retrieve evidence before making any strategic claim."""
    return _get_retriever().retrieve_as_text(query)
```

`@tool` wraps a plain Python function so LangGraph can:
1. Pass its **schema** (name, input types) to the LLM
2. Call it when the LLM decides to use it

**The docstring is critical** — it is literally what the LLM reads to decide WHEN to call this tool. If the docstring is vague, the LLM won't know when to use it. If it's clear and specific, the LLM calls it at the right time.

---

## The Seven Tools

The agent has two categories of tools: general-purpose tools it can use for any query, and section-specific tools that do a focused RAG retrieval for a particular dashboard section.

### General-purpose tools

#### 1. `search_knowledge(query: str)`
The most flexible tool. Accepts any query string, calls `RAGRetriever` which:
- Embeds the query into a 384-dim vector
- Runs cosine similarity search in ChromaDB
- Returns the top 5 most relevant chunks as formatted text

This is the **RAG retrieval** step — the R in RAG. Used for interactive questions in Section 7.

#### 2. `get_sentiment_summary()`
No input needed. Reads all metadata from ChromaDB and counts how many chunks are positive, neutral, and negative. Returns a formatted string like:
```
Sentiment breakdown:
negative: 51 (14%)
neutral: 196 (52%)
positive: 122 (32%)
```

#### 3. `get_topic_summary()`
Same pattern — reads all metadata and counts topic distribution across all stored chunks. Tells the agent what themes dominate the current knowledge base.

### Section-specific tools

These four tools each call `RAGRetriever` with a hardcoded, focused query. The agent doesn't need to decide what to search for — the tool name itself communicates the purpose.

#### 4. `detect_opportunities()`
Searches for: `"NVIDIA strategic opportunities emerging technology new markets partnerships growth expansion"`
Used in pipeline run for Section 3 (Opportunity Monitor).

#### 5. `assess_risks()`
Searches for: `"NVIDIA risks threats competition regulatory legal supply chain challenges negative"`
Used in pipeline run for Section 4 (Risk Monitor).

#### 6. `generate_recommendations()`
Searches for: `"NVIDIA strategic direction CEO priorities competitive positioning market leadership decisions"`
Used in pipeline run for Section 6 (Strategic Recommendations).

#### 7. `get_ceo_briefing_context()`
Searches for: `"NVIDIA recent developments market position news announcements industry trends"`
Used in pipeline run for Section 7 (CEO Briefing).

### Why have separate section-specific tools?

You could achieve the same result by calling `search_knowledge("NVIDIA opportunities growth")`. The reason to have named tools is **explainability**:

When the dashboard shows the agent trace, it displays which tool was called:
```
detect_opportunities()
```
vs.
```
search_knowledge(query="NVIDIA strategic opportunities...")
```

The first version is immediately clear — the professor/reviewer can see that the Opportunity Monitor section used the opportunity detection tool. The tool name documents the purpose without needing to read the query string.

---

## The Singleton Pattern

```python
_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(n_results=5)
    return _retriever
```

`RAGRetriever` loads the embedding model (`all-MiniLM-L6-v2`) when it is first created. Loading a model takes 2-3 seconds. If `search_knowledge` is called 3 times in one agent run, you don't want to reload the model 3 times.

The singleton pattern: create the object once, store it in a module-level variable, return it every time. `global _retriever` tells Python that when we write `_retriever = ...` inside the function, we mean the module-level variable — not a new local one.

---

## The System Prompt

```python
system_prompt = (
    f"You are a strategic intelligence analyst for {TARGET_COMPANY}. "
    "Rules: "
    "1. Always call at least one tool before drawing conclusions. "
    "2. Be specific — name the sources you found (e.g. Barron's, Reddit). "
    "3. Give 3-5 concrete bullet points, no vague generalities. "
    "4. End with a clear, definitive recommendation. "
    "5. Always complete your response — never trail off mid-sentence."
)
```

This is injected at the start of every conversation. It sets the agent's persona and gives explicit instructions on how to behave. The numbered rules push the small model (Qwen2.5:3b) toward specific, grounded answers instead of vague hedging.

---

## The Agent Loop — Step by Step

When `agent.invoke({"messages": [("user", goal)]})` is called:

**Turn 1:**
- LLM reads the goal + system prompt
- Decides: "I need to search the knowledge base first"
- Produces an `AIMessage` with `tool_calls = [{"name": "search_knowledge", "args": {"query": "..."}}]`

**Tool execution:**
- LangGraph sees the tool call, runs `search_knowledge("...")`
- Result comes back as a `ToolMessage` containing the retrieved chunks

**Turn 2:**
- LLM reads the original goal + the ToolMessage result
- Decides: "I have enough evidence" OR "I need another tool"
- If satisfied: produces a final `AIMessage` with `content = "..."` and no tool calls
- If not: another tool call, another loop

**Loop ends** when the LLM produces an `AIMessage` with content and no tool calls.

---

## Parsing the Result

```python
tool_calls_made = []
for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls_made.append({
                "tool": tc["name"],
                "input": tc["args"],
            })

for msg in reversed(result["messages"]):
    if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
        final_answer = msg.content
        break
```

`result["messages"]` is the full message history of the loop:
- `HumanMessage` — the user's goal
- `AIMessage` with tool_calls — each time the LLM called a tool
- `ToolMessage` — each tool result
- `AIMessage` with content — the final answer

We loop forward to collect every tool call (for the trace shown in the dashboard). We loop **backward** to find the last AIMessage with content and no tool calls — that's the final answer.

---

## Why This is Better Than Direct LLM

| Direct LLM | ReAct Agent |
|---|---|
| Answers from training data | Retrieves real evidence first |
| Can hallucinate facts | Every claim is grounded in retrieved chunks |
| Black box | Tool trace shows exactly what evidence was used |
| Fixed answer | Can call multiple tools, iterate on its reasoning |

---

## Practice Questions

Answer these out loud before your viva:

1. What does ReAct stand for and what problem does it solve?

2. What is `create_react_agent` doing — what would you have to write manually if it didn't exist?

3. The docstring on a `@tool` function — who reads it and what do they use it for?

4. Why is the singleton pattern used for `_retriever`? What goes wrong without it?

5. What are the three message types in `result["messages"]` and what does each one represent?

6. Why do we loop **backwards** through the messages to find the final answer?

7. `search_knowledge` calls `RAGRetriever`. What does RAGRetriever actually do — what are the steps?

8. Why do we use Ollama instead of loading Qwen directly with HuggingFace `AutoModelForCausalLM`?

9. The system prompt has 5 numbered rules. Why do we need explicit rules for a small model like Qwen2.5:3b?

10. What is the difference between an `AIMessage` that has `tool_calls` and one that doesn't?

11. We have 7 tools — `detect_opportunities`, `assess_risks`, `generate_recommendations`, `get_ceo_briefing_context`, `search_knowledge`, `get_sentiment_summary`, `get_topic_summary`. What is the difference between the first four and the last three?

12. `detect_opportunities()` takes no arguments. `search_knowledge(query)` takes a query string. Why does the section-specific tool not need a query parameter?

13. The dashboard shows a "tool trace" for each pre-computed section (Opportunities, Risks, etc.). What is the tool trace and why is it useful for the professor reviewing your project?

14. Why is the agent analysis pre-computed to `dashboard_data.json` rather than called live on page load?
