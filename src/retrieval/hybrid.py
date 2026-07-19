"""Hybrid retrieval: BM25 + dense embeddings with RRF fusion."""
import numpy as np
from typing import List
import chromadb

from src.config import settings
from src.ingestion.indexer import DocumentIndexer
from src.models import EvidenceChunk, SourceType


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
    
    Args:
        ranked_lists: List of ranked document ID lists
        k: RRF constant (default 60)
        
    Returns:
        Fused ranking of document IDs
    """
    scores = {}
    
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    """Combines BM25 keyword search with dense vector search."""
    
    def __init__(self):
        self.indexer = DocumentIndexer()
        self.chroma_client = self.indexer.chroma_client
    
    def search(
        self,
        query: str,
        collection_key: str,
        top_k: int = 5
    ) -> List[EvidenceChunk]:
        """
        Perform hybrid search: BM25 + dense → RRF fusion.
        
        Args:
            query: Search query
            collection_key: Collection to search
            top_k: Number of results to return
            
        Returns:
            List of EvidenceChunk objects with relevance scores
        """
        # Get ChromaDB collection
        try:
            collection = self.chroma_client.get_collection(name=collection_key)
        except Exception:
            return []
        
        # Dense retrieval
        query_embedding = self.indexer.embed_query(query)
        dense_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2  # Get more for fusion
        )
        
        dense_ids = dense_results["ids"][0] if dense_results["ids"] else []
        
        # BM25 retrieval
        try:
            bm25_index = self.indexer.load_bm25_index(collection_key)
            
            # Get all documents from collection for BM25 scoring
            all_docs = collection.get(include=["documents", "metadatas"])
            
            if all_docs["documents"]:
                tokenized_query = query.lower().split()
                bm25_scores = bm25_index.get_scores(tokenized_query)
                
                # Get top BM25 results
                bm25_ranked_indices = np.argsort(bm25_scores)[::-1][: top_k * 2]
                bm25_ids = [all_docs["ids"][i] for i in bm25_ranked_indices]
            else:
                bm25_ids = []
        except Exception:
            # Fallback to dense-only if BM25 fails
            bm25_ids = []
        
        # RRF fusion
        if bm25_ids and dense_ids:
            fused_ids = rrf_fuse([dense_ids, bm25_ids], k=60)
        elif dense_ids:
            fused_ids = dense_ids
        elif bm25_ids:
            fused_ids = bm25_ids
        else:
            return []
        
        # Retrieve final results
        final_ids = fused_ids[:top_k]
        
        # Get documents by IDs
        results = collection.get(
            ids=final_ids,
            include=["documents", "metadatas"]
        )
        
        # Build EvidenceChunk objects
        evidence_chunks = []
        
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            content = results["documents"][i]
            
            chunk = EvidenceChunk(
                content=content,
                source=metadata.get("source", "unknown"),
                page=metadata.get("page"),
                section=metadata.get("section"),
                source_type=SourceType.PDF,
                relevance_score=1.0 / (i + 1),  # Simple positional score
                collection=collection_key
            )
            
            evidence_chunks.append(chunk)
        
        return evidence_chunks
