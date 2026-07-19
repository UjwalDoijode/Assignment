"""Safe mathematical computation tool using AST (no eval())."""
import ast
import operator
from langchain.tools import tool


# Safe operations mapping
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def safe_eval(expr: str) -> float:
    """
    Safely evaluate a mathematical expression using AST parsing.
    
    Never uses eval() - only supports basic arithmetic operations.
    
    Args:
        expr: Mathematical expression string
        
    Returns:
        Result as float
        
    Raises:
        ValueError: For unsupported operations or invalid syntax
    """
    def _eval(node):
        if isinstance(node, (ast.Num, ast.Constant)):
            # Handle both old-style Num and new-style Constant nodes
            value = node.n if hasattr(node, 'n') else node.value
            return float(value)
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError(f"Unsupported operation: {op_type.__name__}")
            return SAFE_OPS[op_type](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError(f"Unsupported unary operation: {op_type.__name__}")
            return SAFE_OPS[op_type](_eval(node.operand))
        else:
            raise ValueError(f"Unsupported node type: {ast.dump(node)}")
    
    try:
        tree = ast.parse(expr.strip(), mode='eval')
        return _eval(tree.body)
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")


@tool
def compute(expression: str) -> str:
    """
    Safely compute a mathematical expression.
    
    Supports: +, -, *, /, **, % (modulo), and parentheses.
    Does NOT support: variables, functions, or any code execution.
    
    Args:
        expression: Math expression to evaluate (e.g., "(47.8 - 38.7) / 38.7 * 100")
        
    Returns:
        String representation of the result with explanation
        
    Examples:
        compute("(47.8 - 38.7) / 38.7 * 100")  # Returns percentage change
        compute("120 * 0.85")  # Returns 102.0
    """
    try:
        result = safe_eval(expression)
        return f"Result: {result:.4f} (computed from: {expression})"
    except Exception as e:
        return f"Computation error: {str(e)}"
