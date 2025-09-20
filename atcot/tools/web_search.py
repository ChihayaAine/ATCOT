"""
Web search tool for retrieving online information.
"""

import asyncio
import aiohttp
from typing import Dict, Any, List
from .base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Tool for searching the web and retrieving information."""
    
    def __init__(self, api_key: str = None, search_engine: str = "duckduckgo"):
        super().__init__(
            name="web_search",
            description="Searches the web for information and returns relevant results",
            capabilities=["web_search", "information_retrieval", "real_time_data"]
        )
        self.api_key = api_key
        self.search_engine = search_engine

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute web search."""
        try:
            query = args.get("query", "")
            max_results = args.get("max_results", 5)
            
            if not query:
                return ToolResult(
                    content=None,
                    success=False,
                    error_message="No search query provided"
                )

            # Perform search based on configured engine
            results = await self._perform_search(query, max_results)
            
            return ToolResult(
                content=results,
                success=True,
                confidence=0.8,
                metadata={
                    "query": query,
                    "search_engine": self.search_engine,
                    "result_count": len(results)
                }
            )
            
        except Exception as e:
            return ToolResult(
                content=None,
                success=False,
                error_message=f"Search error: {str(e)}"
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for web search arguments."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }

    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate web search arguments."""
        if "query" not in args or not isinstance(args["query"], str):
            return False
        
        max_results = args.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
            return False
            
        return True

    async def _perform_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform the actual web search."""
        if self.search_engine == "duckduckgo":
            return await self._duckduckgo_search(query, max_results)
        else:
            # Fallback to mock results for demo purposes
            return await self._mock_search(query, max_results)

    async def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo API."""
        try:
            # This is a simplified implementation
            # In a real implementation, you would use the DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        # Process instant answer
                        if data.get("AbstractText"):
                            results.append({
                                "title": data.get("Heading", "Instant Answer"),
                                "snippet": data.get("AbstractText"),
                                "url": data.get("AbstractURL", ""),
                                "source": "DuckDuckGo Instant Answer"
                            })
                        
                        # Process related topics
                        for topic in data.get("RelatedTopics", [])[:max_results-len(results)]:
                            if isinstance(topic, dict) and "Text" in topic:
                                results.append({
                                    "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                    "snippet": topic.get("Text", ""),
                                    "url": topic.get("FirstURL", ""),
                                    "source": "DuckDuckGo Related Topics"
                                })
                        
                        return results[:max_results]
            
            return await self._mock_search(query, max_results)
            
        except Exception:
            # Fallback to mock search
            return await self._mock_search(query, max_results)

    async def _mock_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Mock search results for demonstration purposes."""
        # Simulate network delay
        await asyncio.sleep(0.5)
        
        mock_results = [
            {
                "title": f"Search result 1 for '{query}'",
                "snippet": f"This is a mock search result for the query '{query}'. It contains relevant information about the topic.",
                "url": "https://example.com/result1",
                "source": "Mock Search Engine"
            },
            {
                "title": f"Information about {query}",
                "snippet": f"Additional details and context about {query} can be found here. This mock result provides supplementary information.",
                "url": "https://example.com/result2", 
                "source": "Mock Search Engine"
            },
            {
                "title": f"{query} - Comprehensive Guide",
                "snippet": f"A comprehensive guide covering all aspects of {query}, including background, current status, and future implications.",
                "url": "https://example.com/result3",
                "source": "Mock Search Engine"
            }
        ]
        
        return mock_results[:max_results]
