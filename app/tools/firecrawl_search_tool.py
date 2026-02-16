import os
from typing import List

from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

def firecrawl_search_tool(query: str, num_results: int = 3) -> str:
    app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

    search_result = app.search(query=query, limit=num_results)

    if not search_result.web:
        return "No relevant sources found."

    contents: List[str] = []

    for item in search_result.web:
        page = app.scrape(item.url)

        if not page or not page.markdown:
            continue

        markdown = page.markdown

        contents.append(
            f"Title: {item.title}\n"
            f"Source: {item.url}\n"
            f"{markdown}"
        )

    return "\n\n".join(contents)

#Sample query
response = firecrawl_search_tool("What is database indexing", 2)

print(response)
