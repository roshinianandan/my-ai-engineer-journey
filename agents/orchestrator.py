import time
import ollama
from agents.message_bus import MessageBus, MessageType, Priority
from agents.worker_agents import (
    SearcherAgent, SummarizerAgent,
    CriticAgent, WriterAgent
)
from config import MODEL


class Orchestrator:
    """
    The orchestrator agent that plans and coordinates all worker agents.

    Responsibilities:
    1. Receive complex task from user
    2. Decompose into subtasks
    3. Assign subtasks to appropriate worker agents
    4. Collect and synthesize results
    5. Optionally route to critic for quality check
    6. Produce final output

    The orchestrator never executes tasks directly —
    it only plans, delegates, and synthesizes.
    """

    def __init__(self, bus: MessageBus):
        self.agent_id = "orchestrator"
        self.bus = bus
        self.bus.register_agent(self.agent_id)

        # Initialize all worker agents
        self.workers = {
            "searcher":   SearcherAgent(bus),
            "summarizer": SummarizerAgent(bus),
            "critic":     CriticAgent(bus),
            "writer":     WriterAgent(bus)
        }

        self.task_results = {}

    def decompose_task(self, task: str) -> list:
        """
        Use LLM to break a complex task into ordered subtasks.
        Each subtask is assigned to a specific worker agent.
        """
        available_workers = list(self.workers.keys())

        prompt = f"""Break this research task into subtasks for a multi-agent system.

Task: {task}

Available agents and their capabilities:
- searcher: searches a knowledge base for relevant information
- summarizer: summarizes long text into concise summaries
- critic: evaluates quality and accuracy of content
- writer: writes structured reports from collected findings

Return a JSON array of subtasks in execution order.
Each subtask must have:
- "agent": which agent handles it (must be one of: {available_workers})
- "description": what this subtask does
- "depends_on": list of subtask indices this depends on (empty [] for first tasks)
- "input_key": key name for this task's output
- "params": dict of parameters for the agent

Return ONLY valid JSON. No extra text.

Example format:
[
  {{"agent": "searcher", "description": "Search for X", "depends_on": [], "input_key": "search_results", "params": {{"query": "X", "top_k": 3}}}},
  {{"agent": "summarizer", "description": "Summarize results", "depends_on": [0], "input_key": "summary", "params": {{"style": "concise", "max_sentences": 3}}}}
]"""

        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.2}
            )
            raw = response["message"]["content"].strip()

            # Clean JSON
            import re
            import json
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group())
                return subtasks
        except Exception as e:
            print(f"[Orchestrator] Task decomposition failed: {e}")

        # Fallback: default research pipeline
        return [
            {
                "agent": "searcher",
                "description": f"Search knowledge base for: {task}",
                "depends_on": [],
                "input_key": "search_results",
                "params": {"query": task, "top_k": 4}
            },
            {
                "agent": "summarizer",
                "description": "Summarize search results",
                "depends_on": [0],
                "input_key": "summary",
                "params": {"style": "detailed", "max_sentences": 5}
            },
            {
                "agent": "writer",
                "description": "Write final report",
                "depends_on": [0, 1],
                "input_key": "final_report",
                "params": {
                    "topic": task,
                    "report_type": "research_summary"
                }
            }
        ]

    def execute_subtask(
        self,
        subtask: dict,
        subtask_idx: int,
        context: dict
    ) -> str:
        """Execute a single subtask using the appropriate worker agent."""
        agent_name = subtask["agent"]
        worker = self.workers.get(agent_name)

        if not worker:
            return f"Error: Unknown agent '{agent_name}'"

        # Build task params, injecting context from previous results
        params = subtask.get("params", {}).copy()

        # Inject previous results as needed
        deps = subtask.get("depends_on", [])
        if deps:
            dep_results = []
            for dep_idx in deps:
                dep_key = f"subtask_{dep_idx}"
                if dep_key in context:
                    dep_results.append(context[dep_key])

            # Inject into text/findings params
            if "text" not in params and dep_results:
                params["text"] = "\n\n".join(dep_results)
            if "findings" not in params and dep_results:
                params["findings"] = dep_results

        return worker.execute(params)

    def run(
        self,
        task: str,
        use_critic: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Run the full multi-agent pipeline for a task.

        1. Decompose task into subtasks
        2. Execute subtasks in dependency order
        3. Optionally run critic on final output
        4. Return synthesized result
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  MULTI-AGENT SYSTEM")
        print(f"  Task: {task}")
        print(f"  Agents: {list(self.workers.keys())}")
        print(f"{'='*60}\n")

        # Step 1: Decompose
        print("[Orchestrator] Decomposing task into subtasks...")
        subtasks = self.decompose_task(task)
        print(f"[Orchestrator] Created {len(subtasks)} subtasks:\n")

        for i, st in enumerate(subtasks):
            print(f"  [{i}] {st['agent'].upper()}: {st['description']}")
        print()

        # Step 2: Execute in order
        context = {}
        results = []

        for i, subtask in enumerate(subtasks):
            print(f"\n[Orchestrator] Executing subtask {i}: "
                  f"{subtask['agent'].upper()} — {subtask['description']}")

            result = self.execute_subtask(subtask, i, context)
            context[f"subtask_{i}"] = result
            context[subtask.get("input_key", f"result_{i}")] = result
            results.append({
                "subtask": i,
                "agent": subtask["agent"],
                "description": subtask["description"],
                "result": result
            })

            if verbose:
                print(f"\n  Result preview: {result[:150]}...")

        # Step 3: Get final output
        final_output = results[-1]["result"] if results else "No output generated"

        # Step 4: Critic review (optional)
        critic_review = None
        if use_critic and "critic" not in [r["agent"] for r in results]:
            print(f"\n[Orchestrator] Running critic review...")
            critic_result = self.workers["critic"].execute({
                "content": final_output,
                "original_query": task,
                "criteria": ["accuracy", "completeness", "clarity"]
            })
            critic_review = critic_result
            print(f"\n  Critic review:\n{critic_review}")

        elapsed = round(time.time() - start_time, 2)

        print(f"\n{'='*60}")
        print(f"  FINAL OUTPUT")
        print(f"{'='*60}")
        print(f"\n{final_output}")
        print(f"\n[Completed in {elapsed}s | {len(subtasks)} subtasks | "
              f"{len(self.workers)} agents]")

        return {
            "task": task,
            "final_output": final_output,
            "subtasks_completed": len(subtasks),
            "time_seconds": elapsed,
            "critic_review": critic_review,
            "step_results": results
        }