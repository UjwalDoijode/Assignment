"""Web search tool using Tavily API."""
from langchain.tools import tool
from tavily import TavilyClient

from src.config import settings


# Global Tavily client
_tavily_client = None


def _get_tavily_client() -> TavilyClient:
    """Lazy-load Tavily client singleton."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using Tavily for real-time information.
    
    Use ONLY when the knowledge base does not contain the needed information
    and the query requires current, live data (news, prices, rates, etc.).
    
    Args:
        query: Search query
        max_results: Maximum number of results to return (default 3)
        
    Returns:
        Formatted search results with URLs
        
    Examples:
        web_search("current exchange rate USD to EUR")
        web_search("latest news on electric vehicle batteries")
    """
    try:
        client = _get_tavily_client()
        
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )
        
        if not response.get("results"):
            return "No web results found."
        
        # Format results
        result_parts = ["Web Search Results:", ""]
        
        for i, result in enumerate(response["results"], start=1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            
            result_parts.append(f"{i}. **{title}**")
            result_parts.append(f"   {content[:300]}{'...' if len(content) > 300 else ''}")
            result_parts.append(f"   [Source: {url}]")
            result_parts.append("")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        return f"Web search error: {str(e)}"
