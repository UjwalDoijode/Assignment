"""SQL agent: converts natural language to SQL and executes."""
import json
import sqlite3
from pathlib import Path
from langchain_openai import ChatOpenAI

from src.config import settings
from src.models import ResearchState, SubQuestionResult, EvidenceChunk, SourceType
from src.registry import Registry


def build_nl_to_sql_prompt(
    sub_question: str,
    db_schema: str,
    sample_rows: str
) -> str:
    """Build NL to SQL conversion prompt."""
    
    prompt = f"""SYSTEM:
You are a SQL expert. Convert the natural language question to a valid
SQLite query using ONLY the schema provided.

DATABASE SCHEMA:
{db_schema}

SAMPLE DATA:
{sample_rows}

Rules:
1. Use only tables and columns that exist in the schema above
2. For date filtering, use: strftime('%m', date_column) for month extraction
3. Always alias aggregate columns: SUM(revenue) AS total_revenue
4. Use ROUND(value, 2) for monetary amounts
5. Return ONLY JSON

OUTPUT FORMAT:
{{
  "sql": "SELECT ...",
  "reasoning": "<what this query computes>"
}}

USER:
Convert to SQL: {sub_question}"""
    
    return prompt


def sql_agent_node(state_dict: dict) -> dict:
    """
    SQL agent node: NL → SQL → execute.
    
    For each sql_query sub-question:
    1. Load database schema from registry
    2. Generate SQL query using LLM
    3. Execute query
    4. Store result
    """
    # Handle both dict and ResearchState inputs
    if isinstance(state_dict, ResearchState):
        state = state_dict
    else:
        state = ResearchState(**state_dict)
    
    registry = Registry()
    databases = registry.get_all_databases()
    
    if not databases:
        return state.model_dump()
    
    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        temperature=0.0
    )
    
    # Process sql_query sub-questions
    for sq in state.sub_questions:
        if sq.intent != "sql_query":
            continue
        
        # Use first database (could be enhanced to select specific DB)
        db_entry = databases[0]
        db_path = Path(db_entry["path"])
        
        if not db_path.exists():
            state.sub_results.append(
                SubQuestionResult(
                    sub_question_id=sq.id,
                    question=sq.question,
                    agent_used="sql_agent",
                    sufficient=False,
                    sql_result=f"Error: Database not found: {db_path}"
                )
            )
            continue
        
        # Build prompt
        nl_to_sql_prompt = build_nl_to_sql_prompt(
            sub_question=sq.question,
            db_schema=db_entry["schema"],
            sample_rows=db_entry["sample_rows"]
        )
        
        try:
            # Generate SQL
            response = llm.invoke(nl_to_sql_prompt)
            content = response.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            sql = data.get("sql", "")
            
            # Validate SQL
            if not sql.strip().upper().startswith("SELECT"):
                raise ValueError("Generated query is not a SELECT statement")
            
            # Execute SQL
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            
            # Format result as markdown table
            result_parts = [f"[Source: {db_path.name} - SQL Database]", ""]
            result_parts.append("| " + " | ".join(columns) + " |")
            result_parts.append("| " + " | ".join(["---"] * len(columns)) + " |")
            
            for row in rows[:100]:
                formatted_row = [str(val) if val is not None else "NULL" for val in row]
                result_parts.append("| " + " | ".join(formatted_row) + " |")
            
            if len(rows) > 100:
                result_parts.append(f"\n(Showing first 100 of {len(rows)} rows)")
            
            sql_result_text = "\n".join(result_parts)
            
            # Create evidence chunk
            evidence_chunk = EvidenceChunk(
                content=sql_result_text,
                source=db_path.name,
                source_type=SourceType.SQL,
                relevance_score=1.0
            )
            
            state.sub_results.append(
                SubQuestionResult(
                    sub_question_id=sq.id,
                    question=sq.question,
                    evidence=[evidence_chunk],
                    sql_result=sql_result_text,
                    sufficient=len(rows) > 0,
                    agent_used="sql_agent"
                )
            )
            
        except Exception as e:
            state.sub_results.append(
                SubQuestionResult(
                    sub_question_id=sq.id,
                    question=sq.question,
                    agent_used="sql_agent",
                    sufficient=False,
                    sql_result=f"SQL Error: {str(e)}"
                )
            )
    
    return state.model_dump()
