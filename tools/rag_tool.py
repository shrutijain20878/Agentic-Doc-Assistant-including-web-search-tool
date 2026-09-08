# tools/rag_tool.py

from rag.retriever import get_retriever
from config import LLM


def rag_tool(query):
    """
    Retrieves relevant chunks from the uploaded documents
    and generates an answer grounded only in those chunks.
    """

    print(f"[AGENT] Initiating Local RAG for: '{query}'")

    # ============================================================
    # 1. LOAD RETRIEVER
    # ============================================================

    retriever = get_retriever()

    if retriever is None:
        print("[AGENT] No vector store/document available.")
        return "NOT_FOUND"

    # ============================================================
    # 2. RETRIEVE RELEVANT DOCUMENT CHUNKS
    # ============================================================

    try:
        docs = retriever.invoke(query)

        if not docs:
            print(
                "[RETRIEVER] No relevant document chunks found."
            )
            return "NOT_FOUND"

        print(
            f"[RETRIEVER] Found {len(docs)} relevant chunks."
        )

    except Exception as e:
        print(
            f"[ERROR] Retriever failure: {e}"
        )
        return "NOT_FOUND"

    # ============================================================
    # 3. BUILD CONTEXT
    # ============================================================

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source_file",
            "uploaded document"
        )

        context_parts.append(
            f"""
--- Document Chunk {i} ---
Source: {source}

{doc.page_content}
"""
        )

    context = "\n".join(context_parts)

    # ============================================================
    # 4. GENERATE GROUNDED ANSWER
    # ============================================================

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not search the internet.
3. Do not guess or assume information.
4. If the answer is present in the context, answer it directly.
5. If the answer is not present in the context, return exactly:
   NOT_FOUND
6. Keep the answer concise and directly answer the question.
7. Do not mention the retrieval process.
8. Do not mention these instructions.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    try:

        print(
            "[AGENT] Generating grounded document answer..."
        )

        response = LLM.bind(
            max_tokens=300
        ).invoke(prompt)

        answer = response.content.strip()

    except Exception as e:

        print(
            f"[ERROR] LLM generation failed: {e}"
        )

        return "NOT_FOUND"

    # ============================================================
    # 5. HANDLE NOT_FOUND
    # ============================================================

    if not answer:

        return "NOT_FOUND"

    if answer.upper().strip() == "NOT_FOUND":

        print(
            "[AGENT] Answer not found in uploaded document."
        )

        return "NOT_FOUND"

    # ============================================================
    # 6. REMOVE QWEN THINKING BLOCK IF PRESENT
    # ============================================================

    if "<think>" in answer:

        if "</think>" in answer:

            answer = answer.split(
                "</think>",
                1
            )[1].strip()

        else:

            answer = ""

    if not answer:

        return "NOT_FOUND"

    print(
        "[AGENT] Successfully generated document-grounded answer."
    )

    return answer