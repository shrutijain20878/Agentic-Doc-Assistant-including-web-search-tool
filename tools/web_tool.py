from ddgs import DDGS


def web_tool(query):

    print("\n--- WEB TOOL CALLED ---\n")

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)

        text = ""

        for r in results:
            text += r["body"] + "\n"

    return text