"""Web search agent using Tavily."""
from tavily import TavilyClient

from src.config import settings
from src.models import ResearchState, SubQuestionResult, EvidenceChunk, SourceType


def web_agent_node(state_dict: dict) -> dict:
    """
    Web agent node: performs web searches for real-time information.
    
    Only runs when state.needs_web=True or when sub-question has intent="web_search".
    """
    # Handle both dict and ResearchState inputs
    if isinstance(state_dict, ResearchState):
        state = state_dict
    else:
        state = ResearchState(**state_dict)
    
    # Check if web search is needed
    web_sub_questions = [
        sq for sq in state.sub_questions
        if sq.intent == "web_search"
    ]
    
    if not web_sub_questions and not state.needs_web:
        return state.model_dump()
    
    # Initialize Tavily client
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
    except Exception as e:
        # Skip web search if API key not configured
        for sq in web_sub_questions:
            state.sub_results.append(
                SubQuestionResult(
                    sub_question_id=sq.id,
                    question=sq.question,
                    agent_used="web_agent",
                    sufficient=False
                )
            )
        return state.model_dump()
    
    # Process web_search sub-questions
    for sq in web_sub_questions:
        try:
            response = client.search(
                query=sq.question,
                max_results=3,
                search_depth="basic"
            )
            
            if not response.get("results"):
                state.sub_results.append(
                    SubQuestionResult(
                        sub_question_id=sq.id,
                        question=sq.question,
                        agent_used="web_agent",
                        sufficient=False
                    )
                )
                continue
            
            # Convert results to evidence chunks
            evidence_chunks = []
            
            for result in response["results"]:
                chunk = EvidenceChunk(
                    content=result.get("content", ""),
                    source=result.get("url", ""),
                    source_type=SourceType.WEB,
                    relevance_score=result.get("score", 0.5)
                )
                evidence_chunks.append(chunk)
            
            state.sub_results.append(
                SubQuestionResult(
                    sub_question_id=sq.id,
                    question=sq.question,
                    evidence=evidence_chunks,
                    sufficient=len(evidence_chunks) > 0,
                    agent_used="web_agent"
                )
            )
            
        except Exception as e:
            state.sub_results.append(
                SubQuestionResult(
                    sub_question_id=sq.id,
                    question=sq.question,
                    agent_used="web_agent",
                    sufficient=False
                )
            )
    
    return state.model_dump()
