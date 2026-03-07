# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from tools.rag_tool import rag_tool
from tools.summary_tool import summary_tool
from tools.web_tool import web_tool
from config import LLM
from rag.retriever import get_retriever

# 1. ROUTER NODE
def router(state):
    question = state["question"].lower()
    prompt = f"""
    Analyze the question: "{question}"
    
    Assign to ONE tool:
    - chat: General greetings, definitions, or philosophical questions.
    - rag: Questions about the SPECIFIC contents of uploaded PDF documents.
    - summary: If the user explicitly asks to summarize the uploaded document.
    - web: Questions about real-time events, news, or facts not in the PDFs.

    Return ONLY the tool name.
    """
    raw_response = LLM.invoke(prompt).content.lower().strip()
    
    # Cleaning Logic to prevent KeyError from asterisks or quotes
    tool = raw_response.replace("*", "").replace("'", "").replace('"', "").strip()
    
    valid_tools = ["rag", "summary", "web", "chat"]
    if tool not in valid_tools:
        tool = "chat"
        
    print(f"[AGENT] Routing: '{question}' -> tool: {tool}")
    return {"tool": tool}

# 2. TOOL NODES
def chat_node(state):
    return {"answer": LLM.stream(state["question"])}

def rag_node(state):
    print(f"[AGENT] Attempting RAG for: {state['question']}")
    
    # Get the retriever safely
    retriever = get_retriever()
    
    # FALLBACK: If retriever is missing (None), skip straight to web search
    if retriever is None:
        print("[AGENT] No VectorStore found. Switching to Web Search fallback...")
        return {"answer": "NOT_FOUND"}
        
    # If retriever exists, proceed as normal
    docs = retriever.invoke(state["question"])
    
    if not docs:
        return {"answer": "NOT_FOUND"}
    
    # ... rest of your existing logic (LLM check for context) ...
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Answer using ONLY context. If not found, say NOT_FOUND.\nContext: {context}\nQuestion: {state['question']}"
    
    response = LLM.invoke(prompt).content.strip()
    if "NOT_FOUND" in response.upper():
        return {"answer": "NOT_FOUND"}
        
    return {"answer": LLM.stream(prompt)}

def summary_node(state):
    return {"answer": summary_tool(state["question"])}

def web_node(state):
    content = web_tool(state["question"])
    prompt = f"The documents didn't mention this. Based on the web info, answer the question: {state['question']}\n\nInfo:\n{content}"
    return {"answer": LLM.stream(prompt)}

# 3. CONDITIONAL ROUTING LOGIC
def route_tools(state):
    """Initial routing from the router node."""
    return state["tool"]

def decide_after_rag(state):
    if state["answer"] == "NOT_FOUND":
        return "web"  # This now catches missing DBs AND missing info
    return "end"

# 4. BUILD THE GRAPH
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("router", router)
builder.add_node("rag", rag_node)
builder.add_node("summary", summary_node)
builder.add_node("web", web_node)
builder.add_node("chat", chat_node)

# Define Flow
builder.set_entry_point("router")

# Router -> Tools
builder.add_conditional_edges(
    "router",
    route_tools,
    {
        "rag": "rag",
        "summary": "summary",
        "web": "rag",
        "chat": "chat",
    }
)

# --- THE KEY FIX: RAG Fallback ---
# If RAG is successful, it goes to END. If NOT_FOUND, it goes to WEB.
builder.add_conditional_edges(
    "rag",
    decide_after_rag,
    {
        "web": "web",
        "end": END
    }
)

# Other nodes always go to END
builder.add_edge("summary", END)
builder.add_edge("web", END)
builder.add_edge("chat", END)

# Compile
graph = builder.compile()
