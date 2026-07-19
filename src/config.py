"""Configuration management using pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  # Ignore extra fields in .env
    )
    
    # LLM Configuration
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    
    # Tavily Configuration
    tavily_api_key: str
    
    # LangSmith Configuration (optional)
    langsmith_api_key: str | None = None
    langchain_project: str = "valiance-research"
    langchain_tracing_v2: bool = True
    
    # Storage Paths
    chroma_path: Path = Path("./chroma_db")
    bm25_path: Path = Path("./bm25_indexes")
    sqlite_path: Path = Path("./corpus/databases")
    corpus_pdf_path: Path = Path("./corpus/pdfs")
    registry_path: Path = Path("./registry.json")
    trace_output_dir: Path = Path("./traces")
    
    # Retrieval Configuration
    max_rag_iterations: int = 3
    embedding_model: str = "all-MiniLM-L6-v2"
    skip_verification: bool = False
    direct_retrieval: bool = False  # Skip agentic query formulation loop
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.bm25_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.mkdir(parents=True, exist_ok=True)
        self.corpus_pdf_path.mkdir(parents=True, exist_ok=True)
        self.trace_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure LangSmith if API key provided
        if self.langsmith_api_key:
            import os
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.langchain_tracing_v2).lower()
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project


# Global settings instance
settings = Settings()
