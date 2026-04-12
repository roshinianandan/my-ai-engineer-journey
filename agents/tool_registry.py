import math
from tools.weather_tool import get_weather, WEATHER_TOOL_SCHEMA
from tools.calculator_tool import calculate, CALCULATOR_TOOL_SCHEMA
from tools.search_tool import search_knowledge_base, SEARCH_TOOL_SCHEMA


def summarize_text(text: str, max_words: int = 100) -> dict:
    """Summarize text to a given word count."""
    words = text.split()
    if len(words) <= max_words:
        return {"summary": text, "original_words": len(words)}
    truncated = " ".join(words[:max_words]) + "..."
    return {
        "summary": truncated,
        "original_words": len(words),
        "summary_words": max_words
    }


SUMMARIZE_TOOL_SCHEMA = {
    "name": "summarize_text",
    "description": "Summarize a long piece of text into a shorter version. Use when the user wants a summary or when retrieved text is too long.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to summarize"
            },
            "max_words": {
                "type": "integer",
                "description": "Maximum words in summary. Default is 100.",
                "default": 100
            }
        },
        "required": ["text"]
    }
}


AGENT_TOOLS = {
    "get_weather": {
        "function": get_weather,
        "schema": WEATHER_TOOL_SCHEMA,
        "description": "Get current weather for any city"
    },
    "calculate": {
        "function": calculate,
        "schema": CALCULATOR_TOOL_SCHEMA,
        "description": "Perform mathematical calculations safely"
    },
    "search_knowledge_base": {
        "function": search_knowledge_base,
        "schema": SEARCH_TOOL_SCHEMA,
        "description": "Search local AI knowledge base"
    },
    "summarize_text": {
        "function": summarize_text,
        "schema": SUMMARIZE_TOOL_SCHEMA,
        "description": "Summarize long text into shorter version"
    }
}


def get_tool(name: str):
    """Get a tool function by name."""
    return AGENT_TOOLS.get(name)


def execute_tool(name: str, params: dict) -> dict:
    """Execute a tool and return its result."""
    tool = get_tool(name)
    if not tool:
        return {"error": f"Unknown tool: {name}. Available: {list(AGENT_TOOLS.keys())}"}
    try:
        return tool["function"](**params)
    except Exception as e:
        return {"error": f"Tool failed: {str(e)}"}


def tool_descriptions_for_prompt() -> str:
    """Format tool descriptions for injection into agent prompt."""
    lines = []
    for name, tool in AGENT_TOOLS.items():
        schema = tool["schema"]
        params = schema["parameters"]["properties"]
        required = schema["parameters"].get("required", [])
        param_desc = ", ".join(
            f"{k}({'required' if k in required else 'optional'})"
            for k in params
        )
        lines.append(f"- {name}({param_desc}): {schema['description']}")
    return "\n".join(lines)