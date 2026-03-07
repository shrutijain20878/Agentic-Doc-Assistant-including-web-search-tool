# rag_tool.py
from rag.retriever import get_retriever
from config import LLM

def rag_tool(query):
    retriever = get_retriever()
    if retriever is None:
        print("[PRODUCTION LOG] Vector store not found. Redirecting to Web.")
        return "NOT_FOUND"
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"[ERROR] Retriever failed: {e}")
        return "NOT_FOUND"
    
    # If no documents are found at all by the retriever
    if not docs:
        return "NOT_FOUND"
    
    context = "\n".join([d.page_content for d in docs])
    
    prompt = f"""
    Answer the following question using ONLY the provided context.
    If the context does not contain the answer, strictly respond with the exact word: NOT_FOUND
    
    Context:
    {context}
    
    Question: {query}
    """
    
    # Use invoke to check if the answer exists
    response = LLM.invoke(prompt).content.strip()
    
    # Clean the response to check for our keyword
    if "NOT_FOUND" in response.upper():
        return "NOT_FOUND"

    # If it found the data, return the stream
    return LLM.stream(prompt)