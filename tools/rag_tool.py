# rag_tool.py
from rag.retriever import get_retriever
from config import LLM

def rag_tool(query):
    print(f"[AGENT] Initiating Local RAG for: '{query}'")
    
    retriever = get_retriever()
    if retriever is None:
        print("[AGENT] Status: Vector store empty/missing. Routing to Web Fallback.")
        return "NOT_FOUND"

    try:
        # 1. Retrieve the Documents
        docs = retriever.invoke(query)
        if not docs:
            print("[RETRIEVER] Status: No matching documents found in Local DB.")
            return "NOT_FOUND"
        
        print(f"[RETRIEVER] Success: Found {len(docs)} relevant context chunks.")

    except Exception as e:
        print(f"[ERROR] Retriever failure: {e}")
        return "NOT_FOUND"
    
    # 2. Prepare Context for the "Verification" check
    context = "\n".join([d.page_content for d in docs])
    
    verification_prompt = f"""
    Answer the following question using ONLY the provided context.
    If the context does NOT contain the answer, strictly respond with the exact word: NOT_FOUND
    
    Context:
    {context}
    
    Question: {query}
    """
    
    # 3. The "Gatekeeper" Check: Does the context actually answer the question?
    print("[AGENT] Verifying context relevance...")
    check_response = LLM.invoke(verification_prompt).content.strip()
    
    if "NOT_FOUND" in check_response.upper():
        print("[AGENT] Status: Context irrelevant. Escalating to Web Search.")
        return "NOT_FOUND"

    # 4. Success: If we are here, we have the answer!
    print("[AGENT] Status: Context verified. Generating grounded response.")
    return LLM.stream(verification_prompt)