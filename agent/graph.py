# agent/graph.py

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from tools.rag_tool import rag_tool
from tools.summary_tool import summary_tool
from tools.web_tool import web_tool
from config import LLM
from rag.retriever import get_retriever


# ============================================================
# 1. ROUTER NODE
# ============================================================

def router(state):
    question = state["question"]

    prompt = f"""
You are the routing agent for an Agentic RAG Document Assistant.

Choose exactly ONE route.

RAG:
Use when the user asks for information that may be present
in an uploaded document.

SUMMARY:
Use when the user asks to summarize, summarize briefly,
give an overview, or condense an uploaded document.

WEB:
Use when the user asks for current, live, recent, or external
information.

CHAT:
Use for greetings, casual conversation, or general questions
that do not require a document or current web information.

Examples:

hi -> CHAT
hello -> CHAT
what is RAG -> CHAT
how many years of experience does Shruti have -> RAG
what are Shruti's skills -> RAG
what projects are mentioned -> RAG
summarize -> SUMMARY
summarize the uploaded document -> SUMMARY
give me an overview of the resume -> SUMMARY
latest current affairs -> WEB
what is today's news -> WEB

USER QUESTION:
{question}

Return ONLY one of:
RAG
SUMMARY
WEB
CHAT
"""

    try:

        response = LLM.bind(
            max_tokens=10
        ).invoke(prompt)

        raw = response.content.strip().upper()

        print(f"[ROUTER RAW] {raw}")

        if raw == "RAG":
            tool = "rag"

        elif raw == "SUMMARY":
            tool = "summary"

        elif raw == "WEB":
            tool = "web"

        elif raw == "CHAT":
            tool = "chat"

        else:
            print(
                f"[ROUTER] Unexpected response: {raw}"
            )
            tool = "chat"

        print(
            f"[AGENT] Routing: '{question}' -> tool: {tool}"
        )

        return {
            "tool": tool
        }

    except Exception as e:

        print(
            f"[ROUTER] Error: {e}"
        )

        return {
            "tool": "chat"
        }

# ============================================================
# 2. CHAT NODE
# ============================================================

def chat_node(state):

    prompt = f"""
You are a helpful AI Assistant.

Respond politely and concisely to:

{state["question"]}
"""

    return {
        "answer": LLM.stream(prompt)
    }


# ============================================================
# 3. RAG NODE
# ============================================================

def rag_node(state):

    print(
        f"[AGENT] RAG question: {state['question']}"
    )

    response = rag_tool(
        state["question"]
    )

    return {
        "answer": response
    }


# ============================================================
# 4. SUMMARY NODE
# ============================================================

def summary_node(state):

    print(
        f"[AGENT] Summarizing document: "
        f"{state['question']}"
    )

    retriever = get_retriever()

    if retriever is None:

        return {
            "answer": "No document has been uploaded yet."
        }

    try:

        docs = retriever.invoke(
            state["question"]
        )

        if not docs:

            return {
                "answer": "I couldn't find any document content to summarize."
            }

        context = "\n".join(
            doc.page_content
            for doc in docs
        )

        summary = summary_tool(
            state["question"],
            context
        )

        return {
            "answer": summary
        }

    except Exception as e:

        print(
            f"[SUMMARY] Error: {e}"
        )

        return {
            "answer": "I couldn't summarize the document."
        }


# ============================================================
# 5. WEB NODE
# ============================================================

def web_node(state):

    print(
        f"[AGENT] Web Search: "
        f"{state['question']}"
    )

    search_content = web_tool(
        state["question"]
    )

    if "SEARCH_FAILED" in search_content:

        prompt = f"""
The user asked:

{state["question"]}

Web search is currently unavailable.

Answer using your general knowledge.
Do not claim that you performed a web search.
"""

    else:

        prompt = f"""
Answer the user's question using the following web
search results.

WEB SEARCH RESULTS:
{search_content}

USER QUESTION:
{state["question"]}

Give a concise and useful answer.
"""

    return {
        "answer": LLM.stream(prompt)
    }


# ============================================================
# 6. ROUTING
# ============================================================

def route_tools(state):

    return state["tool"]


def decide_after_rag(state):

    if state["answer"] == "NOT_FOUND":

        print(
            "[AGENT] RAG could not find the answer."
        )

        return "web"

    return "end"


# ============================================================
# 7. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(
    AgentState
)


# Add nodes
builder.add_node(
    "router",
    router
)

builder.add_node(
    "rag",
    rag_node
)

builder.add_node(
    "summary",
    summary_node
)

builder.add_node(
    "web",
    web_node
)

builder.add_node(
    "chat",
    chat_node
)


# Entry point
builder.set_entry_point(
    "router"
)


# Router → Tool
builder.add_conditional_edges(
    "router",
    route_tools,
    {
        "rag": "rag",
        "summary": "summary",
        "web": "web",
        "chat": "chat",
    }
)


# RAG → END / WEB fallback
builder.add_conditional_edges(
    "rag",
    decide_after_rag,
    {
        "web": "web",
        "end": END,
    }
)


# Other tools → END
builder.add_edge(
    "summary",
    END
)

builder.add_edge(
    "web",
    END
)

builder.add_edge(
    "chat",
    END
)


# Compile
graph = builder.compile()