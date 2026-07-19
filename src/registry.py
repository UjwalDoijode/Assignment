"""Registry management for collection metadata and routing."""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.config import settings


class Registry:
    """Manages the collection registry (registry.json)."""
    
    def __init__(self):
        self.registry_path = settings.registry_path
        self.data = self._load()
    
    def _load(self) -> dict:
        """Load registry from disk or return empty structure."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        return {
            "generated_at": datetime.now().isoformat(),
            "collections": {},
            "sql_databases": []
        }
    
    def save(self):
        """Save registry to disk."""
        self.data["generated_at"] = datetime.now().isoformat()
        
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def add_collection(
        self,
        collection_key: str,
        display_name: str,
        description: str,
        keywords: list[str],
        source_files: list[str],
        chunk_count: int,
        centroid_embedding: np.ndarray
    ):
        """Add a collection to the registry."""
        self.data["collections"][collection_key] = {
            "display_name": display_name,
            "description": description,
            "keywords": keywords,
            "source_files": source_files,
            "chunk_count": chunk_count,
            "centroid_embedding": centroid_embedding.tolist()
        }
    
    def add_sql_database(
        self,
        db_path: str,
        schema: str,
        sample_rows: str,
        description: str
    ):
        """Add a SQL database to the registry."""
        self.data["sql_databases"].append({
            "path": db_path,
            "schema": schema,
            "sample_rows": sample_rows,
            "description": description
        })
    
    def get_collection(self, collection_key: str) -> Optional[dict]:
        """Get collection metadata by key."""
        return self.data["collections"].get(collection_key)
    
    def get_all_collections(self) -> dict:
        """Get all collections."""
        return self.data["collections"]
    
    def get_all_databases(self) -> list[dict]:
        """Get all SQL databases."""
        return self.data["sql_databases"]
    
    def route_by_centroid(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.25,
        top_k: int = 2
    ) -> list[str]:
        """
        Route a query to collections using centroid similarity.
        
        Args:
            query_embedding: Query vector
            threshold: Minimum similarity score
            top_k: Maximum collections to return
            
        Returns:
            List of collection keys sorted by similarity
        """
        scores = {}
        
        for coll_key, meta in self.data["collections"].items():
            centroid = np.array(meta["centroid_embedding"])
            
            # Cosine similarity
            dot = np.dot(query_embedding, centroid)
            norm = np.linalg.norm(query_embedding) * np.linalg.norm(centroid)
            similarity = dot / norm if norm > 0 else 0.0
            
            if similarity > threshold:
                scores[coll_key] = similarity
        
        # Sort by score descending and take top_k
        sorted_collections = sorted(
            scores.items(),
            key=lambda x: -x[1]
        )
        
        return [k for k, v in sorted_collections[:top_k]]
    
    def build_orchestrator_prompt_context(self) -> dict:
        """
        Build dynamic prompt context for the orchestrator.
        
        Returns:
            Dict with collection_descriptions and sql_descriptions
        """
        collection_desc = []
        for key, meta in self.data["collections"].items():
            collection_desc.append(
                f"- **{key}** ({meta['display_name']}): {meta['description']}"
            )
        
        sql_desc = []
        for db in self.data["sql_databases"]:
            sql_desc.append(
                f"- **{Path(db['path']).name}**: {db['description']}"
            )
        
        return {
            "collection_descriptions": "\n".join(collection_desc) if collection_desc else "None",
            "sql_descriptions": "\n".join(sql_desc) if sql_desc else "None"
        }
