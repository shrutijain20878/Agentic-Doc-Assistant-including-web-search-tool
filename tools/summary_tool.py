from rag.retriever import get_retriever
from config import LLM

def summary_tool(query):
    retriever = get_retriever()
    # For summary, we pull more chunks
    retriever.k = 15
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Summarize these specific document excerpts:\n{context}"
    return LLM.stream(prompt)
    # answer = ""
    # for token in LLM.stream_invoke(prompt):
    #     answer += token
    #     yield token