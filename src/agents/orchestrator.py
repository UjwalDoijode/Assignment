"""Orchestrator agent: decomposes questions into sub-questions."""
import json
from langchain_openai import ChatOpenAI

from src.config import settings
from src.models import ResearchState, SubQuestion
from src.registry import Registry


# Input classification categories
INPUT_GREETING = "greeting"
INPUT_OFF_TOPIC = "off_topic"
INPUT_RESEARCH = "research"

GREETING_PATTERNS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "sup", "yo", "what's up", "how are you",
    "how do you do", "nice to meet", "hola", "namaste",
]


def classify_input(question: str) -> str:
    """Classify user input as greeting, off-topic, or research question."""
    q = question.strip().lower().rstrip("!?.")
    
    # Check for greeting patterns
    for pattern in GREETING_PATTERNS:
        if q == pattern or q.startswith(pattern + " ") or q.startswith(pattern + ","):
            return INPUT_GREETING
    
    # Very short inputs (< 3 words) that aren't questions are likely not research
    words = q.split()
    if len(words) <= 2 and not any(w in q for w in ["what", "how", "why", "when", "where", "who", "which", "compare", "list", "show", "tell", "explain", "describe"]):
        # Could be a greeting variant or too vague
        if len(words) == 1 and words[0] not in ["help"]:
            return INPUT_GREETING
    
    return INPUT_RESEARCH


def build_orchestrator_prompt(original_question: str, registry: Registry) -> str:
    """Build orchestrator prompt dynamically from registry."""
    
    context = registry.build_orchestrator_prompt_context()
    
    prompt = f"""SYSTEM:
You are a research orchestrator for a private knowledge base system.
Your job is to decompose a complex user question into focused sub-questions,
each answerable from exactly ONE source.

You must ONLY answer questions that can be researched using the available
knowledge sources listed below. If the question is unrelated to any available
source, return an empty sub_questions array.

Available knowledge sources:
{context['collection_descriptions']}

Available structured databases:
{context['sql_descriptions']}

Rules:
1. Break the question into 1-4 sub-questions. Keep each sub-question atomic.
2. Assign each sub-question an intent:
   - "kb_lookup"   → answered from a document collection
   - "sql_query"   → answered by querying a database
   - "web_search"  → requires live internet data not present in any document
   - "compute"     → pure arithmetic on values already retrieved
3. For "kb_lookup", set target_collection to the most relevant collection key.
4. Set needs_web=true ONLY when the question requires real-time data
   (exchange rates, live news, current prices) that cannot exist in stored documents.
5. Add "compute" sub-questions only when explicit calculation is needed
   (percentage change, difference, conversion) AFTER data is retrieved.
6. If the question is completely unrelated to the available knowledge sources,
   return an empty sub_questions array with reasoning explaining why.
7. Return ONLY valid JSON. No text outside the JSON block.

OUTPUT FORMAT:
{{
  "sub_questions": [
    {{
      "id": "sq_1",
      "question": "<focused sub-question>",
      "intent": "kb_lookup",
      "target_collection": "<collection_key>",
      "needs_compute": false,
      "depends_on": []
    }}
  ],
  "needs_web": false,
  "reasoning": "<brief explanation of decomposition>"
}}

USER:
{original_question}"""
    
    return prompt


def orchestrator_node(state_dict: dict) -> dict:
    """
    Orchestrator node: decompose question into sub-questions.
    
    1. Classify input (greeting/off-topic/research)
    2. For research questions: build prompt from registry, call LLM,
       parse response into SubQuestion objects.
    3. For greetings/off-topic: return friendly response without LLM call.
    """
    # Handle both dict and ResearchState inputs
    if isinstance(state_dict, ResearchState):
        state = state_dict
    else:
        state = ResearchState(**state_dict)
    
    # --- GUARDRAIL: Classify input ---
    input_type = classify_input(state.original_question)
    
    if input_type == INPUT_GREETING:
        state.final_brief = (
            "Hello! I'm a research assistant designed to answer questions about your "
            "knowledge base. I can search documents, query databases, and perform calculations.\n\n"
            "Try asking me a question like:\n"
            "- \"What products are available and their specifications?\"\n"
            "- \"What was the revenue in Q4?\"\n"
            "- \"What is the warranty policy?\"\n\n"
            "How can I help you today?"
        )
        state.is_complete = True
        state.verification_passed = True
        state.verification_notes = "Greeting response — no verification needed"
        return state.model_dump()
    
    # Load registry
    registry = Registry()
    
    # Build prompt
    prompt = build_orchestrator_prompt(state.original_question, registry)
    
    # Call LLM
    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        temperature=0.0
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Extract JSON (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # LLM returned prose instead of JSON — likely off-topic or adversarial query.
            # Treat as unanswerable rather than surfacing a raw parse error.
            state.final_brief = (
                "I wasn't able to find relevant information for that question in the "
                "knowledge base. I'm designed to answer research questions about the "
                "documents and databases that have been indexed.\n\n"
                "Try asking something like:\n"
                "- \"What products are available and their specifications?\"\n"
                "- \"What was the revenue in Q4?\"\n"
                "- \"What is the warranty policy?\"\n"
            )
            state.is_complete = True
            state.verification_passed = True
            state.verification_notes = "No research needed — query outside knowledge base scope"
            return state.model_dump()
        
        # Build SubQuestion objects
        sub_questions = [
            SubQuestion(**sq) for sq in data.get("sub_questions", [])
        ]
        
        # If orchestrator returned empty sub-questions, provide helpful response
        if not sub_questions:
            reasoning = data.get("reasoning", "")
            state.final_brief = (
                "I wasn't able to find relevant information for that question in the "
                "knowledge base.\n\n"
                f"Reason: {reasoning}\n\n" if reasoning else
                "I wasn't able to match your question to any available knowledge source.\n\n"
                "Try asking a question related to the indexed documents or databases."
            )
            state.is_complete = True
            state.verification_passed = True
            state.verification_notes = "No matching knowledge sources found"
            return state.model_dump()
        
        # Update state
        state.sub_questions = sub_questions
        state.needs_web = data.get("needs_web", False)
        
        return state.model_dump()
        
    except Exception as e:
        state.error = f"Orchestrator error: {str(e)}"
        state.final_brief = "An error occurred while processing your question. Please try again."
        state.is_complete = True
        return state.model_dump()
