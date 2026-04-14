import time
import argparse
import ollama
from agents.custom_tools import LANGCHAIN_TOOLS
from agents.agent_memory import AgentMemoryManager
from config import MODEL


# Build a tool lookup dictionary
TOOL_MAP = {tool.name: tool for tool in LANGCHAIN_TOOLS}


def get_tool_descriptions() -> str:
    """Format all tool descriptions for the prompt."""
    return "\n".join(
        f"- {tool.name}: {tool.description}" for tool in LANGCHAIN_TOOLS
    )


def call_llm(messages: list, temperature: float = 0.2) -> str:
    """Call Ollama directly."""
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        stream=False,
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def parse_action(response: str) -> dict:
    """
    Parse the LLM response for Action and Action Input.
    Returns dict with thought, action, action_input, final_answer.
    """
    import re

    result = {
        "thought": "",
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    # Extract Thought
    thought_match = re.search(
        r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL
    )
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Extract Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)$", response, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # Extract Action
    action_match = re.search(
        r"Action:\s*(.+?)(?=Action Input:|$)", response, re.DOTALL
    )
    if action_match:
        result["action"] = action_match.group(1).strip()

    # Extract Action Input
    input_match = re.search(
        r"Action Input:\s*(.+?)(?=Thought:|Action:|Final Answer:|Observation:|$)",
        response,
        re.DOTALL,
    )
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    return result


def run_agent(
    task: str,
    max_iterations: int = 8,
    verbose: bool = True,
    history: list | None = None,
) -> str:
    """
    Run a ReAct-style agent loop using direct Ollama calls.

    Args:
        task:           The user task / question.
        max_iterations: Hard cap on reasoning steps.
        verbose:        Print step-by-step trace.
        history:        Optional prior conversation messages (dicts with
                        role/content) to prepend for memory continuity.
    """
    tool_descriptions = get_tool_descriptions()
    tool_names = list(TOOL_MAP.keys())

    system_prompt = f"""You are a helpful AI assistant with access to tools.

Available tools:
{tool_descriptions}

Use this EXACT format:

Thought: think about what to do
Action: tool_name (must be one of: {tool_names})
Action Input: the input for the tool
Observation: (tool result will appear here)
... (repeat as needed)
Thought: I now have the final answer
Final Answer: your complete answer here

Rules:
- Always write Thought before Action
- Action must be exactly one tool name from the list
- End with Final Answer when done
- Never skip the Thought step"""

    # Build initial message list, optionally including prior history
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if history:
        for entry in history:
            role = "user" if entry["role"] == "human" else "assistant"
            messages.append({"role": role, "content": entry["content"]})

    messages.append({"role": "user", "content": f"Task: {task}"})

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LANGCHAIN AGENT (Direct)")
        print(f"  Task: {task}")
        print(f"  Tools: {tool_names}")
        print(f"{'='*60}\n")

    for i in range(max_iterations):
        if verbose:
            print(f"\n[Step {i+1}/{max_iterations}]")

        response = call_llm(messages)

        if verbose:
            print(f"LLM Response:\n{response}\n")

        parsed = parse_action(response)

        # ── Final answer ──────────────────────────────────────────────
        if parsed["final_answer"]:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  FINAL ANSWER")
                print(f"{'='*60}")
                print(f"\n{parsed['final_answer']}")
            return parsed["final_answer"]

        # ── Tool execution ────────────────────────────────────────────
        action = (parsed.get("action") or "").strip()
        action_input = (parsed.get("action_input") or "").strip()

        if not action:
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": "Please continue. Use the format: Thought / Action / Action Input or Final Answer.",
                }
            )
            continue

        if action not in TOOL_MAP:
            observation = (
                f"Error: Tool '{action}' not found. "
                f"Available tools: {tool_names}"
            )
        else:
            try:
                if verbose:
                    print(f"🔧 Calling tool: {action}")
                    print(f"   Input: {action_input}")

                observation = TOOL_MAP[action].invoke(action_input)

                if verbose:
                    print(f"   Result: {str(observation)[:200]}...")

            except Exception as e:
                observation = f"Tool error: {str(e)}"

        messages.append({"role": "assistant", "content": response})
        messages.append(
            {
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue with your next Thought.",
            }
        )

    # Max iterations reached — ask for best-effort final answer
    final_response = call_llm(
        messages
        + [
            {
                "role": "user",
                "content": "Based on everything gathered, give your Final Answer now.",
            }
        ]
    )
    return final_response


def run_research_agent(topic: str) -> dict:
    """Run a structured research task through the agent."""
    print(f"\n{'='*60}")
    print(f"  RESEARCH AGENT")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    start = time.time()

    task = (
        f'Research the topic: "{topic}"\n'
        "1. Search the knowledge base for relevant information\n"
        "2. Summarize the key findings\n"
        "3. Write a structured research report"
    )

    answer = run_agent(task, verbose=True)
    elapsed = round(time.time() - start, 2)

    print(f"\n  Time: {elapsed}s")
    return {"topic": topic, "report": answer, "time": elapsed}


def interactive_agent_session():
    """Interactive agent session with sliding-window memory."""
    memory = AgentMemoryManager(window_size=5)

    print(f"\n🤖 LangChain Agent (Direct Mode)")
    print(f"   Model: {MODEL}")
    print(f"   Tools: {list(TOOL_MAP.keys())}")
    print(f"   Memory: last {memory.window_size} exchanges")
    print("\n   Commands: 'quit' | 'history' | 'clear' | 'stats'")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        match user_input.lower():
            case "quit":
                print("👋 Goodbye!")
                break
            case "history":
                memory.show_history()
                continue
            case "clear":
                memory.clear()
                continue
            case "stats":
                print(memory.stats())
                continue

        start = time.time()
        answer = run_agent(
            user_input,
            verbose=True,
            history=memory.get_history(),
        )
        elapsed = round(time.time() - start, 2)

        memory.add_interaction(user_input, answer)

        print(f"\n🤖 Agent: {answer}")
        print(f"\n   [{elapsed}s]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain Agent")
    parser.add_argument("--research", type=str, help="Research a topic")
    parser.add_argument("--chat", action="store_true", help="Interactive session with memory")
    parser.add_argument("--task", type=str, help="Run a single task")
    args = parser.parse_args()

    if args.research:
        run_research_agent(args.research)
    elif args.task:
        answer = run_agent(args.task, verbose=True)
        print(f"\n🤖 Final Answer: {answer}")
    elif args.chat:
        interactive_agent_session()
    else:
        parser.print_help()