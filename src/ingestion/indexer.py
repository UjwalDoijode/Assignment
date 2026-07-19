"""Indexing pipeline: chunk, embed, upsert to ChromaDB and BM25."""
import pickle
import numpy as np
from pathlib import Path
from typing import Iterator
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingestion.pdf_parser import PDFChunk


class DocumentIndexer:
    """Handles embedding, vector storage, and BM25 indexing."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
    
    def index_document(
        self,
        collection_key: str,
        chunks: Iterator[PDFChunk],
        source_filename: str
    ) -> tuple[int, np.ndarray]:
        """
        Index a document: embed chunks, upsert to ChromaDB, build BM25, compute centroid.
        
        Args:
            collection_key: Unique key for the collection (slug from filename)
            chunks: Iterator of PDFChunk objects
            source_filename: Original filename for metadata
            
        Returns:
            Tuple of (chunk_count, centroid_embedding)
        """
        # Collect chunks into memory (needed for BM25 and centroid)
        chunk_list = list(chunks)
        
        if not chunk_list:
            return 0, np.array([])
        
        # Extract text and metadata
        texts = [chunk.content for chunk in chunk_list]
        metadatas = [
            {
                "source": source_filename,
                "page": chunk.page,
                "section": chunk.section,
                "collection": collection_key
            }
            for chunk in chunk_list
        ]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Get or create ChromaDB collection
        collection = self.chroma_client.get_or_create_collection(
            name=collection_key,
            metadata={"source": source_filename}
        )
        
        # Upsert to ChromaDB
        ids = [f"{collection_key}_{i}" for i in range(len(texts))]
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        # Build BM25 index
        tokenized_corpus = [text.lower().split() for text in texts]
        bm25_index = BM25Okapi(tokenized_corpus)
        
        # Save BM25 index
        bm25_path = settings.bm25_path / f"{collection_key}.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        return len(chunk_list), centroid
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        return self.embedding_model.encode(query, convert_to_numpy=True)
    
    def load_bm25_index(self, collection_key: str) -> BM25Okapi:
        """Load a BM25 index from disk."""
        bm25_path = settings.bm25_path / f"{collection_key}.pkl"
        
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
        
        with open(bm25_path, "rb") as f:
            return pickle.load(f)
