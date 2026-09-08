from ddgs import DDGS


def web_tool(query: str):
    """
    Search the web using DuckDuckGo and return a small amount
    of relevant context for the LLM.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "SEARCH_FAILED: No results found."

        context = "\n".join(
            [
                f"Title: {result.get('title', '')}\n"
                f"Snippet: {result.get('body', '')}"
                for result in results
            ]
        )

        return context

    except Exception as e:
        print(f"[ERROR] Search Tool Error: {e}")
        return "SEARCH_FAILED: Connectivity issue."