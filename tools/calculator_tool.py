import math
import ast
import operator

# Tool schema
CALCULATOR_TOOL_SCHEMA = {
    "name": "calculate",
    "description": "Perform mathematical calculations. Use this for any arithmetic, algebra, percentages, statistics, or math operations the user needs. Do not try to calculate in your head — always use this tool for numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate. Examples: '2 + 2', '15 * 8', 'sqrt(144)', '(100 - 20) / 4', '2 ** 10'"
            }
        },
        "required": ["expression"]
    }
}

# Safe operators — prevents code injection
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "pow": math.pow,
    "factorial": math.factorial,
}


def _safe_eval(node):
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function not allowed: {func_name}")
        args = [_safe_eval(arg) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.
    Uses AST parsing to prevent code injection — never uses eval() directly.
    """
    try:
        expression = expression.strip()
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)

        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 6)

        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero", "success": False}
    except Exception as e:
        return {"expression": expression, "error": str(e), "success": False}