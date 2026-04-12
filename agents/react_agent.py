import json
import re
import time
from agents.base_agent import BaseAgent, AgentStep
from agents.tool_registry import execute_tool, tool_descriptions_for_prompt, AGENT_TOOLS


REACT_SYSTEM_PROMPT = """You are an intelligent AI agent that solves tasks step by step.

You have access to these tools:
{tool_descriptions}

Use this EXACT format for every response:

Thought: [Think about what you need to do to answer the question]
Action: [tool_name OR "final_answer"]
Action Input: {{"param1": "value1", "param2": "value2"}}

Rules:
- Always start with Thought
- Use Action: final_answer when you have enough information to answer
- For final_answer, use Action Input: {{"answer": "your complete answer here"}}
- Never skip the Thought step
- Be precise with tool names — they must match exactly
- If a tool fails, try a different approach

Available tool names: {tool_names}"""


class ReActAgent(BaseAgent):
    """
    ReAct Agent — Reason + Act loop built from scratch.

    The agent follows this loop:
    1. Receive task
    2. Think about what to do (Thought)
    3. Decide on an action (Action + Action Input)
    4. Execute the action (Observation)
    5. Go back to step 2 unless action is final_answer
    6. Stop if max_steps is reached
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation = []

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current tool descriptions."""
        return REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions_for_prompt(),
            tool_names=list(AGENT_TOOLS.keys())
        )

    def _parse_react_response(self, response: str) -> tuple:
        """
        Parse the LLM response for Thought, Action, and Action Input.
        Returns (thought, action, action_input_dict)
        """
        thought = ""
        action = ""
        action_input = {}

        # Extract Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=Action:|$)", response,
            re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(
            r"Action:\s*(.+?)(?=Action Input:|$)", response,
            re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).strip()

        # Extract Action Input
        input_match = re.search(
            r"Action Input:\s*(\{.+?\})", response,
            re.DOTALL | re.IGNORECASE
        )
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to clean and parse
                raw = input_match.group(1)
                raw = re.sub(r"'", '"', raw)
                try:
                    action_input = json.loads(raw)
                except Exception:
                    action_input = {}

        return thought, action, action_input

    def _execute_action(self, action: str, action_input: dict) -> str:
        """Execute a tool action and return the observation string."""
        if action == "final_answer":
            return action_input.get("answer", "")

        result = execute_tool(action, action_input)
        return json.dumps(result, indent=2)

    def run(self, task: str) -> dict:
        """
        Run the ReAct loop for a given task.
        Returns a dict with the final answer, steps taken, and metadata.
        """
        self.reset()
        self.start_time = time.time()

        self.log(f"\n{'='*60}")
        self.log(f"  REACT AGENT")
        self.log(f"  Task: {task}")
        self.log(f"  Max steps: {self.max_steps}")
        self.log(f"{'='*60}")

        # Build initial messages
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"}
        ]

        final_answer = None
        step_num = 0

        while step_num < self.max_steps:
            step_num += 1
            step = AgentStep(step_num)

            self.log(f"\n[Step {step_num}/{self.max_steps}]")

            # Get LLM response
            response = self.call_llm(messages)
            self.log(f"Raw response:\n{response}", prefix="  ")

            # Parse the response
            thought, action, action_input = self._parse_react_response(response)
            step.thought = thought
            step.action = action
            step.action_input = action_input

            if not action:
                self.log("  [No action detected — treating as final answer]")
                final_answer = response
                step.observation = "No structured action found"
                self.steps.append(step)
                break

            # Check for final answer
            if action.lower() in ["final_answer", "final answer"]:
                final_answer = action_input.get("answer", response)
                step.observation = "Task complete"
                self.steps.append(step)
                self.log_step(step)
                break

            # Execute the tool
            self.log(f"  Action: {action}")
            self.log(f"  Input: {json.dumps(action_input)}")
            observation = self._execute_action(action, action_input)
            step.observation = observation
            self.steps.append(step)
            self.log_step(step)

            # Add to conversation for next iteration
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue your reasoning."
            })

        else:
            # Max steps reached
            self.log(f"\n[Max steps ({self.max_steps}) reached — generating best answer]")
            messages.append({
                "role": "user",
                "content": "You have reached the maximum number of steps. Based on everything you have gathered, give your best final answer now."
            })
            final_answer = self.call_llm(messages, temperature=0.3)

        elapsed = round(time.time() - self.start_time, 2)

        result = {
            "task": task,
            "answer": final_answer,
            "steps_taken": len(self.steps),
            "max_steps": self.max_steps,
            "time_seconds": elapsed,
            "step_log": self.get_step_log()
        }

        self.log(f"\n{'='*60}")
        self.log(f"  FINAL ANSWER")
        self.log(f"{'='*60}")
        self.log(f"\n{final_answer}")
        self.log(f"\n  Steps taken: {len(self.steps)} | Time: {elapsed}s")

        return result


def run_react_demo():
    """Run the ReAct agent on a set of demo tasks."""
    agent = ReActAgent(max_steps=6, verbose=True)

    tasks = [
        "What is the weather in Chennai and what is 25 percent of the temperature in Celsius?",
        "Search for information about RAG systems and summarize what you find in 50 words.",
        "Calculate the compound interest on Rs 50000 at 8 percent annual rate for 3 years.",
    ]

    for task in tasks:
        result = agent.run(task)
        agent.save_log(f"react_{len(agent.steps)}_steps.json")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ReAct Agent")
    parser.add_argument("--task",  type=str, help="Single task to run")
    parser.add_argument("--demo",  action="store_true", help="Run demo tasks")
    parser.add_argument("--steps", type=int, default=6, help="Max steps")
    parser.add_argument("--chat",  action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.demo:
        run_react_demo()
    elif args.task:
        agent = ReActAgent(max_steps=args.steps, verbose=True)
        result = agent.run(args.task)
        agent.save_log("react_latest.json")
    elif args.chat:
        agent = ReActAgent(max_steps=args.steps, verbose=True)
        print("\n🤖 ReAct Agent — Interactive Mode")
        print("   Type any task. The agent will reason through it step by step.")
        print("   Type 'quit' to exit.\n")
        while True:
            task = input("Task: ").strip()
            if not task:
                continue
            if task.lower() == "quit":
                break
            result = agent.run(task)
            agent.save_log("react_latest.json")
    else:
        parser.print_help()