"""Pydantic models for the multi-agent research system."""
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of evidence source."""
    PDF = "pdf"
    SQL = "sql"
    WEB = "web"
    COMPUTED = "computed"


class EvidenceChunk(BaseModel):
    """A single piece of retrieved evidence."""
    content: str
    source: str                     # filename or URL
    page: Optional[int] = None
    section: Optional[str] = None   # "text" or "table_N"
    source_type: SourceType = SourceType.PDF
    relevance_score: float = 0.0
    collection: Optional[str] = None


class SubQuestion(BaseModel):
    """A decomposed sub-question to be answered by a specific agent."""
    id: str                         # "sq_1", "sq_2" ...
    question: str
    intent: Literal["kb_lookup", "sql_query", "web_search", "compute"]
    target_collection: Optional[str] = None
    needs_compute: bool = False
    depends_on: list[str] = Field(default_factory=list)


class SubQuestionResult(BaseModel):
    """Result from answering a sub-question."""
    sub_question_id: str
    question: str
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    sql_result: Optional[str] = None
    computed_result: Optional[str] = None
    sufficient: bool = False
    iterations: int = 0
    agent_used: str = ""


class ResearchState(BaseModel):
    """Complete state for the research graph."""
    original_question: str = ""
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    needs_web: bool = False
    sub_results: list[SubQuestionResult] = Field(default_factory=list)
    final_brief: str = ""
    citations: list[dict] = Field(default_factory=list)
    verification_passed: bool = False
    verification_notes: str = ""
    error: Optional[str] = None
    is_complete: bool = False
