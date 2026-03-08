import asyncio
import nest_asyncio
from duckduckgo_search import DDGS # Updated Import

# Fix for Streamlit's event loop
nest_asyncio.apply()

def web_tool(query: str):
    """
    Asynchronous web search using the latest DDGS interface.
    """
    async def fetch_results():
        try:
            # The DDGS class now handles async natively in newer versions
            with DDGS() as ddgs:
                # We wrap the generator in a list for the result
                results = [r for r in ddgs.text(query, max_results=3)]
                
                if not results:
                    return "SEARCH_FAILED: No results found."
                
                context = "\n".join([
                    f"Title: {r['title']}\nSnippet: {r['body']}" 
                    for r in results
                ])
                return context
        except Exception as e:
            print(f"[ERROR] Search Tool Error: {e}")
            return "SEARCH_FAILED: Connectivity issue."

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(fetch_results())
    except Exception as e:
        # Fallback if the loop is already running and nest_asyncio failed
        return "SEARCH_FAILED: Runtime error."