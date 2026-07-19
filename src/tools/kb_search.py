"""Knowledge base search tool with centroid-based routing."""
from langchain.tools import tool
from typing import List

from src.registry import Registry
from src.retrieval.hybrid import HybridRetriever
from src.ingestion.indexer import DocumentIndexer
from src.models import EvidenceChunk


# Global instances (initialized on first use)
_registry = None
_retriever = None
_indexer = None


def _get_registry() -> Registry:
    """Lazy-load registry singleton."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def _get_retriever() -> HybridRetriever:
    """Lazy-load retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_indexer() -> DocumentIndexer:
    """Lazy-load indexer singleton."""
    global _indexer
    if _indexer is None:
        _indexer = DocumentIndexer()
    return _indexer


@tool
def kb_search(query: str, collection: str = None, top_k: int = 5) -> str:
    """
    Search the knowledge base using hybrid BM25 + dense retrieval.
    
    If collection is specified, searches that collection directly.
    Otherwise, uses centroid-based routing to find the best collection(s).
    
    Args:
        query: Natural language search query
        collection: Optional specific collection key to search
        top_k: Number of results to return (default 5)
        
    Returns:
        Formatted string of evidence chunks with citations
        
    Examples:
        kb_search("battery life of M-100 model")
        kb_search("Q4 revenue", collection="financial_reports")
    """
    registry = _get_registry()
    retriever = _get_retriever()
    indexer = _get_indexer()
    
    # Determine which collections to search
    if collection:
        collections_to_search = [collection]
    else:
        # Use centroid routing
        query_embedding = indexer.embed_query(query)
        collections_to_search = registry.route_by_centroid(
            query_embedding,
            threshold=0.25,
            top_k=2
        )
    
    if not collections_to_search:
        return "No relevant collections found for this query."
    
    # Search each collection and aggregate results
    all_evidence: List[EvidenceChunk] = []
    
    for coll_key in collections_to_search:
        evidence = retriever.search(query, coll_key, top_k=top_k)
        all_evidence.extend(evidence)
    
    if not all_evidence:
        return "No results found in the knowledge base."
    
    # Sort by relevance score
    all_evidence.sort(key=lambda x: x.relevance_score, reverse=True)
    
    # Take top results
    top_evidence = all_evidence[:top_k]
    
    # Format results
    result_parts = []
    
    for i, chunk in enumerate(top_evidence, start=1):
        citation = f"[Source: {chunk.source}"
        if chunk.page:
            citation += f", p.{chunk.page}"
        citation += "]"
        
        result_parts.append(
            f"{i}. {chunk.content[:500]}{'...' if len(chunk.content) > 500 else ''}\n"
            f"   {citation}\n"
            f"   Collection: {chunk.collection}\n"
        )
    
    return "\n".join(result_parts)
