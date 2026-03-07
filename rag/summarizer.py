from config import LLM

def summarize_docs(docs):

    text = "\n".join([d.page_content for d in docs])

    prompt = f"""
Summarize this document:

{text}
"""

    response = LLM.invoke(prompt)

    return response.content