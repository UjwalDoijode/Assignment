"""LangGraph orchestration: StateGraph with conditional routing."""
from typing import Literal
from langgraph.graph import StateGraph, END

from src.models import ResearchState
from src.agents.orchestrator import orchestrator_node
from src.agents.rag_agent import rag_agent_node
from src.agents.sql_agent import sql_agent_node
from src.agents.web_agent import web_agent_node
from src.agents.synthesis import synthesis_node


def route_after_orchestrator(state_dict: dict) -> list[str]:
    """
    Conditional routing after orchestrator.
    
    Returns list of agent names to execute based on sub-question intents.
    """
    # Handle both dict and ResearchState inputs
    if isinstance(state_dict, ResearchState):
        state = state_dict
    else:
        state = ResearchState(**state_dict)
    
    if not state.sub_questions:
        return ["synthesis"]
    
    agents_needed = set()
    
    for sq in state.sub_questions:
        if sq.intent == "kb_lookup":
            agents_needed.add("rag_agent")
        elif sq.intent == "sql_query":
            agents_needed.add("sql_agent")
        elif sq.intent == "web_search":
            agents_needed.add("web_agent")
    
    # If needs_web flag is set, include web agent
    if state.needs_web:
        agents_needed.add("web_agent")
    
    # If no agents needed, go directly to synthesis
    if not agents_needed:
        return ["synthesis"]
    
    return list(agents_needed)


def build_research_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for multi-agent research.
    
    Flow:
    - orchestrator → (rag_agent | sql_agent | web_agent) → synthesis → END
    """
    # Create graph
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("web_agent", web_agent_node)
    graph.add_node("synthesis", synthesis_node)
    
    # Set entry point
    graph.set_entry_point("orchestrator")
    
    # Conditional fan-out after orchestrator
    # Note: LangGraph's conditional_edges with multiple targets requires
    # a function that returns a single string, not a list
    # We'll use a simple approach where we check which agents are needed
    # and route sequentially
    
    def router(state_dict: dict) -> Literal["rag_agent", "sql_agent", "web_agent", "synthesis"]:
        """Route to first needed agent or synthesis."""
        agents = route_after_orchestrator(state_dict)
        
        if "rag_agent" in agents:
            return "rag_agent"
        elif "sql_agent" in agents:
            return "sql_agent"
        elif "web_agent" in agents:
            return "web_agent"
        else:
            return "synthesis"
    
    graph.add_conditional_edges(
        "orchestrator",
        router,
        {
            "rag_agent": "rag_agent",
            "sql_agent": "sql_agent",
            "web_agent": "web_agent",
            "synthesis": "synthesis",
        }
    )
    
    # Route from agents to next agent or synthesis
    def after_rag(state_dict: dict) -> Literal["sql_agent", "web_agent", "synthesis"]:
        """Route after RAG agent."""
        if isinstance(state_dict, ResearchState):
            state = state_dict
        else:
            state = ResearchState(**state_dict)
        remaining_intents = {sq.intent for sq in state.sub_questions}
        processed_agents = {r.agent_used for r in state.sub_results}
        
        if "sql_query" in remaining_intents and "sql_agent" not in processed_agents:
            return "sql_agent"
        elif ("web_search" in remaining_intents or state.needs_web) and "web_agent" not in processed_agents:
            return "web_agent"
        else:
            return "synthesis"
    
    def after_sql(state_dict: dict) -> Literal["web_agent", "synthesis"]:
        """Route after SQL agent."""
        if isinstance(state_dict, ResearchState):
            state = state_dict
        else:
            state = ResearchState(**state_dict)
        remaining_intents = {sq.intent for sq in state.sub_questions}
        processed_agents = {r.agent_used for r in state.sub_results}
        
        if ("web_search" in remaining_intents or state.needs_web) and "web_agent" not in processed_agents:
            return "web_agent"
        else:
            return "synthesis"
    
    graph.add_conditional_edges(
        "rag_agent",
        after_rag,
        {
            "sql_agent": "sql_agent",
            "web_agent": "web_agent",
            "synthesis": "synthesis",
        }
    )
    
    graph.add_conditional_edges(
        "sql_agent",
        after_sql,
        {
            "web_agent": "web_agent",
            "synthesis": "synthesis",
        }
    )
    
    # Web agent always goes to synthesis
    graph.add_edge("web_agent", "synthesis")
    
    # Synthesis is the end
    graph.add_edge("synthesis", END)
    
    return graph.compile()
