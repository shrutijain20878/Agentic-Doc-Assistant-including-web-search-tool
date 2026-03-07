from config import LLM

def run_qa(query, docs):

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}
"""

    response = LLM.invoke(prompt)

    return response.content