🤖 Agentic-Doc-Assistant
An Intelligent Knowledge Orchestrator built with LangGraph, ChromaDB, and Groq.

The Agentic-Doc-Assistant is a state-aware AI agent designed to bridge the gap between static documents and the real-time web. Unlike traditional RAG systems, this assistant uses a "RAG-First" logic: it prioritizes your uploaded data but autonomously pivots to a global web search if the information is missing.

🚀 Key Features
💬 Intelligent Chat: Handles natural greetings and general conversation.

📚 Context-Aware RAG: Deep-dive Q&A using ChromaDB vector embeddings.

⚡ Smart Summarization: Generates comprehensive overviews of uploaded PDFs.

🌐 Autonomous Web Fallback: If the vector store returns NOT_FOUND, the agent automatically triggers a live web search to find the answer.

🧠 State Management: Powered by LangGraph to ensure reliable transitions between tools.

🏗️ System Architecture
This project uses a directed acyclic graph (DAG) to manage the decision-making process.
graph TD
    A[User Input] --> B{Router Node}
    B -- Greeting --> C[Chat Node]
    B -- Summarize --> D[Summary Node]
    B -- Factual Query --> E[RAG Node]
    
    E -- Data Found --> F[Generate Answer]
    E -- NOT_FOUND --> G[Web Search Node]
    
    G --> F
    C --> H[End]
    D --> H
    F --> H

🛠️ Tech Stack
Orchestration: LangGraph

LLM: Groq (Llama 3.1 8B/70B)

Vector Database: ChromaDB

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Frontend: Streamlit

Search Integration: DuckDuckGo / Tavily API