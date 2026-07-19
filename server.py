"""FastAPI server for Multi-Agent Deep Research System."""
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.graph import build_research_graph
from src.models import ResearchState
from src.registry import Registry
from src.config import settings


# Request/Response models
class ResearchRequest(BaseModel):
    question: str
    mode: Literal["ultra_fast", "balanced", "full_quality"] = "balanced"


class ResearchResponse(BaseModel):
    question: str
    brief: str
    verification_passed: bool
    verification_notes: str
    mode: str
    duration_seconds: float
    trace_file: str


class HealthResponse(BaseModel):
    status: str
    registry_loaded: bool
    collections: int


# FastAPI app
app = FastAPI(title="Multi-Agent Deep Research API")

# CORS middleware - allow all origins (dev/demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry
registry = None


@app.on_event("startup")
async def startup_event():
    """Load registry on startup."""
    global registry
    
    registry_path = settings.registry_path
    
    if registry_path.exists():
        try:
            registry = Registry()
            collections = registry.get_all_collections()
            print(f"✓ Registry loaded: {len(collections)} collections")
            for coll_key in collections.keys():
                print(f"  - {coll_key}")
        except Exception as e:
            print(f"⚠ Warning: Failed to load registry: {e}")
            registry = None
    else:
        print(f"⚠ Warning: Registry not found at {registry_path}. Run ingest.py first.")
        registry = None


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Execute research query with specified performance mode.
    
    Modes:
    - ultra_fast: Skip query formulation, skip verification, max 1 iteration
    - balanced: Use query formulation, skip verification, max 1 iteration
    - full_quality: Full agentic loop, enable verification, max 3 iterations
    """
    # Set environment variables based on mode
    mode_config = {
        "ultra_fast": {
            "DIRECT_RETRIEVAL": "true",
            "SKIP_VERIFICATION": "true",
            "MAX_RAG_ITERATIONS": "1"
        },
        "balanced": {
            "DIRECT_RETRIEVAL": "false",
            "SKIP_VERIFICATION": "true",
            "MAX_RAG_ITERATIONS": "1"
        },
        "full_quality": {
            "DIRECT_RETRIEVAL": "false",
            "SKIP_VERIFICATION": "false",
            "MAX_RAG_ITERATIONS": "3"
        }
    }
    
    config = mode_config[request.mode]
    for key, value in config.items():
        os.environ[key] = value
    
    # Reload settings to pick up new env vars
    from src.config import Settings
    current_settings = Settings()
    
    # Build graph
    try:
        graph = build_research_graph()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build graph: {str(e)}")
    
    # Execute research
    start_time = time.time()
    
    try:
        initial_state = ResearchState(original_question=request.question)
        result = graph.invoke(initial_state.model_dump())
        duration = time.time() - start_time
        
        # Extract result fields
        final_brief = result.get("final_brief", "")
        verification_passed = result.get("verification", {}).get("passed", True)
        verification_notes = result.get("verification", {}).get("notes", "")
        
        # Find trace file (most recent in traces/)
        trace_dir = settings.trace_output_dir
        if trace_dir.exists():
            traces = sorted(trace_dir.glob("trace_*.json"), key=os.path.getmtime, reverse=True)
            trace_file = str(traces[0]) if traces else ""
        else:
            trace_file = ""
        
        return ResearchResponse(
            question=request.question,
            brief=final_brief,
            verification_passed=verification_passed,
            verification_notes=verification_notes,
            mode=request.mode,
            duration_seconds=round(duration, 2),
            trace_file=trace_file
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    registry_loaded = registry is not None
    collection_count = len(registry.get_all_collections()) if registry else 0
    
    return HealthResponse(
        status="ok",
        registry_loaded=registry_loaded,
        collections=collection_count
    )


@app.get("/collections")
async def get_collections():
    """List all indexed document collections."""
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Registry not found. Run ingest.py first."
        )
    
    collections_list = []
    all_collections = registry.get_all_collections()
    for key, metadata in all_collections.items():
        collections_list.append({
            "name": key,
            "description": metadata["description"],
            "chunk_count": metadata["chunk_count"],
            "source_files": metadata["source_files"]
        })
    
    return collections_list


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
