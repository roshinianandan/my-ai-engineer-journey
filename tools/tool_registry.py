import json
from tools.weather_tool import get_weather, WEATHER_TOOL_SCHEMA
from tools.calculator_tool import calculate, CALCULATOR_TOOL_SCHEMA
from tools.search_tool import search_knowledge_base, SEARCH_TOOL_SCHEMA


# All available tools registered in one place
TOOLS = {
    "get_weather": {
        "function": get_weather,
        "schema": WEATHER_TOOL_SCHEMA
    },
    "calculate": {
        "function": calculate,
        "schema": CALCULATOR_TOOL_SCHEMA
    },
    "search_knowledge_base": {
        "function": search_knowledge_base,
        "schema": SEARCH_TOOL_SCHEMA
    }
}


def get_all_schemas() -> list:
    """Return all tool schemas for injection into the LLM prompt."""
    return [tool["schema"] for tool in TOOLS.values()]


def execute_tool(tool_name: str, parameters: dict) -> dict:
    """
    Execute a registered tool by name with given parameters.
    Returns the tool's result or an error if the tool is not found.
    """
    if tool_name not in TOOLS:
        return {"error": f"Tool '{tool_name}' not found. Available: {list(TOOLS.keys())}"}

    try:
        fn = TOOLS[tool_name]["function"]
        result = fn(**parameters)
        return result
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


def format_schemas_for_prompt() -> str:
    """Format all tool schemas as a readable string for the LLM prompt."""
    schemas = get_all_schemas()
    lines = ["Available tools:\n"]

    for schema in schemas:
        lines.append(f"Tool: {schema['name']}")
        lines.append(f"Description: {schema['description']}")
        lines.append("Parameters:")
        for param, details in schema["parameters"]["properties"].items():
            required = param in schema["parameters"].get("required", [])
            lines.append(f"  - {param} ({'required' if required else 'optional'}): {details['description']}")
        lines.append("")

    return "\n".join(lines)