from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from vectorstore import get_retriever
from dotenv import load_dotenv

load_dotenv()


tavily=TavilySearch(max_results=3,search_depth='basic',topic='general')

@tool
def web_search_tool(query: str) -> str:
    """
    Executes a real-time web search using Tavily and returns 
    structured results with title, content snippet, and URL.

    Args:
        query (str): The search query string provided by the user.

    Returns:
        str: A formatted string containing the search results 
             (title, content, and URL). If no results are found, 
             it returns 'No results found'. In case of failure, 
             it returns an error message prefixed with 'WEB_ERROR::'.
    """
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"
    
@tool
def rag_search_tool(query: str) -> str:
    """
    Retrieves the most relevant knowledge base chunks (Top-K) for a given query.

    Args:
        query (str): The user query string used to search the knowledge base.

    Returns:
        str: A concatenated string of the retrieved document chunks. 
             Returns an empty string if no relevant chunks are found. 
             In case of failure, returns an error message prefixed with 'RAG_ERROR::'.
    """
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query) 
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"