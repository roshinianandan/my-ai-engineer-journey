from langchain.tools import tool
from tools.weather_tool import get_weather
from tools.calculator_tool import calculate
from tools.search_tool import search_knowledge_base
import ollama
from config import MODEL


@tool
def weather_tool(city: str) -> str:
    """
    Get the current weather for any city.
    Input should be just the city name like Chennai or London.
    Returns temperature, conditions, humidity and wind speed.
    """
    result = get_weather(city)
    return (
        f"Weather in {result['city']}: "
        f"{result['temperature']}{result['unit']}, "
        f"{result['condition']}, "
        f"Humidity: {result['humidity_percent']}%, "
        f"Wind: {result['wind_kph']} kph"
    )


@tool
def calculator_tool(expression: str) -> str:
    """
    Perform mathematical calculations safely.
    Input should be a valid math expression like 2 + 2 or sqrt(144).
    Supports: +, -, *, /, **, %, sqrt, abs, round, log, sin, cos, tan, factorial.
    Always use this for any numeric computation.
    """
    result = calculate(expression)
    if result["success"]:
        return f"Result of {result['expression']} = {result['result']}"
    else:
        return f"Calculation error: {result['error']}"


@tool
def knowledge_base_tool(query: str) -> str:
    """
    Search the local knowledge base for information about AI, machine learning,
    deep learning, Python tools, RAG systems, and related technical topics.
    Input should be a clear search query.
    Returns the most relevant text chunks found.
    """
    result = search_knowledge_base(query, top_k=3)
    if not result["found"]:
        return "No relevant information found in the knowledge base."

    output_parts = [f"Found {result['num_results']} relevant results:\n"]
    for i, r in enumerate(result["results"], 1):
        output_parts.append(
            f"[{i}] Source: {r['source']} | Score: {r['relevance_score']}\n"
            f"{r['text'][:300]}\n"
        )
    return "\n".join(output_parts)


@tool
def summarizer_tool(text: str) -> str:
    """
    Summarize a long piece of text into 3 clear sentences.
    Use this when you have retrieved information that is too long.
    Input should be the full text you want summarized.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Summarize this in exactly 3 sentences:\n\n{text}"
        }],
        stream=False,
        options={"temperature": 0.3}
    )
    return response["message"]["content"]


@tool
def report_writer_tool(content: str) -> str:
    """
    Format gathered information into a structured research report.
    Input should be all the raw information and findings collected.
    Returns a well-structured report with Executive Summary, Key Findings, and Conclusion.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"""Format the following into a structured research report.
Include: Executive Summary, Key Findings, and Conclusion.

Information:
{content}

Report:"""
        }],
        stream=False,
        options={"temperature": 0.4}
    )
    return response["message"]["content"]


LANGCHAIN_TOOLS = [
    weather_tool,
    calculator_tool,
    knowledge_base_tool,
    summarizer_tool,
    report_writer_tool
]