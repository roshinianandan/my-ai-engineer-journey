import json
import ollama
import argparse
from config import MODEL
from tools.tool_registry import execute_tool, format_schemas_for_prompt, TOOLS

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

{tool_descriptions}

When you need to use a tool, respond with EXACTLY this format and nothing else:
TOOL_CALL: {{
  "tool": "tool_name",
  "parameters": {{
    "param1": "value1"
  }}
}}

After receiving the tool result, use it to give a helpful, natural response to the user.
If no tool is needed, just answer directly.
Think carefully about which tool is most appropriate for each request."""


def detect_tool_call(response: str) -> dict | None:
    """
    Check if the LLM response contains a tool call.
    Returns the parsed tool call dict or None if no tool call detected.
    """
    if "TOOL_CALL:" not in response:
        return None

    try:
        start = response.find("TOOL_CALL:") + len("TOOL_CALL:")
        json_str = response[start:].strip()

        # Find the end of the JSON object
        brace_count = 0
        end = 0
        for i, char in enumerate(json_str):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        json_str = json_str[:end]
        return json.loads(json_str)

    except Exception as e:
        print(f"  [Tool call parse error: {e}]")
        return None


def run_tool_agent(user_input: str, conversation_history: list) -> str:
    """
    Single turn of the tool agent:
    1. Send user input to LLM with tool descriptions
    2. If LLM requests a tool, execute it
    3. Send tool result back to LLM for final response
    4. Return the final answer
    """
    tool_descriptions = format_schemas_for_prompt()
    system_content = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    messages = [{"role": "system", "content": system_content}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})

    # First LLM call — decide if tool is needed
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        stream=False,
        options={"temperature": 0.2}
    )
    first_response = response["message"]["content"]

    tool_call = detect_tool_call(first_response)

    if tool_call:
        tool_name = tool_call.get("tool", "")
        parameters = tool_call.get("parameters", {})

        print(f"\n  🔧 Using tool: {tool_name}")
        print(f"     Parameters: {parameters}")

        # Execute the tool
        tool_result = execute_tool(tool_name, parameters)
        print(f"     Result: {json.dumps(tool_result, indent=2)[:200]}...")

        # Second LLM call — generate response using tool result
        messages.append({"role": "assistant", "content": first_response})
        messages.append({
            "role": "user",
            "content": f"Tool result: {json.dumps(tool_result)}\n\nPlease give a helpful response to the user based on this result."
        })

        final_response = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=False,
            options={"temperature": 0.5}
        )
        return final_response["message"]["content"]

    return first_response


def interactive_agent():
    """Run an interactive tool-using agent session."""
    print(f"\n🤖 Tool Agent Ready")
    print(f"   Model: {MODEL}")
    print(f"   Tools: {', '.join(TOOLS.keys())}")
    print("\n   Try asking:")
    print("   - What is the weather in Coimbatore?")
    print("   - What is 15% of 8500?")
    print("   - What is sqrt(2401)?")
    print("   - Search for information about RAG systems")
    print("   - Calculate compound interest: 10000 * (1 + 0.08) ** 5")
    print("\n   Type 'quit' to exit\n")
    print("-" * 55)

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break

        response = run_tool_agent(user_input, conversation_history)
        print(f"\n🤖 Assistant: {response}")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool-Using AI Agent")
    parser.add_argument("--query", type=str, help="Single query to test")
    args = parser.parse_args()

    if args.query:
        print(f"\nQuery: {args.query}")
        result = run_tool_agent(args.query, [])
        print(f"\n🤖 Response: {result}")
    else:
        interactive_agent()