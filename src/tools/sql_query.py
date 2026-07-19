"""SQL query tool for structured databases."""
import sqlite3
from pathlib import Path
from langchain.tools import tool

from src.config import settings
from src.registry import Registry


# Global registry instance
_registry = None


def _get_registry() -> Registry:
    """Lazy-load registry singleton."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


@tool
def sql_query(sql: str, database: str = None) -> str:
    """
    Execute a SQL query on a registered SQLite database.
    
    Only SELECT queries are allowed. The query must use tables and columns
    that exist in the database schema.
    
    Args:
        sql: SQL SELECT query to execute
        database: Optional specific database filename (e.g., "sales.sqlite").
                 If not specified, uses the first available database.
        
    Returns:
        Query results formatted as a table or error message
        
    Examples:
        sql_query("SELECT SUM(revenue) as total FROM sales WHERE quarter='Q4'")
        sql_query("SELECT model, COUNT(*) FROM transactions GROUP BY model", database="sales.sqlite")
    """
    # Validate query is SELECT only
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
    
    # Prevent dangerous operations
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    if any(keyword in sql_upper for keyword in dangerous_keywords):
        return f"Error: Query contains forbidden keyword. Only SELECT is allowed."
    
    # Get database path
    registry = _get_registry()
    databases = registry.get_all_databases()
    
    if not databases:
        return "Error: No databases registered. Run ingest.py first."
    
    # Select database
    if database:
        db_entry = next(
            (db for db in databases if Path(db["path"]).name == database),
            None
        )
        if not db_entry:
            available = [Path(db["path"]).name for db in databases]
            return f"Error: Database '{database}' not found. Available: {', '.join(available)}"
    else:
        db_entry = databases[0]
    
    db_path = Path(db_entry["path"])
    
    if not db_path.exists():
        return f"Error: Database file not found: {db_path}"
    
    # Execute query
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        
        if not rows:
            return "Query executed successfully but returned no rows."
        
        # Format results as markdown table
        result_parts = [f"[Source: {db_path.name} - SQL Database]", ""]
        
        # Header
        result_parts.append("| " + " | ".join(columns) + " |")
        result_parts.append("| " + " | ".join(["---"] * len(columns)) + " |")
        
        # Rows (limit to 100 for safety)
        for row in rows[:100]:
            formatted_row = [str(val) if val is not None else "NULL" for val in row]
            result_parts.append("| " + " | ".join(formatted_row) + " |")
        
        if len(rows) > 100:
            result_parts.append(f"\n(Showing first 100 of {len(rows)} rows)")
        
        return "\n".join(result_parts)
        
    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Error executing query: {str(e)}"
