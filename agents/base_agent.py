import json
import ollama
from datetime import datetime
from config import MODEL


class AgentStep:
    """Represents one step in the agent's reasoning loop."""

    def __init__(self, step_num: int):
        self.step_num = step_num
        self.thought = ""
        self.action = None
        self.action_input = None
        self.observation = None
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "timestamp": self.timestamp
        }

    def __str__(self) -> str:
        parts = [f"\n--- Step {self.step_num} ---"]
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.action_input:
            parts.append(f"Input: {json.dumps(self.action_input)}")
        if self.observation:
            obs_str = json.dumps(self.observation)
            if len(obs_str) > 200:
                obs_str = obs_str[:200] + "..."
            parts.append(f"Observation: {obs_str}")
        return "\n".join(parts)


class BaseAgent:
    """
    Base class for all agents in this project.
    Handles LLM calls, step logging, and common utilities.
    """

    def __init__(
        self,
        model: str = MODEL,
        max_steps: int = 6,
        temperature: float = 0.2,
        verbose: bool = True
    ):
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose
        self.steps: list[AgentStep] = []
        self.start_time = None

    def call_llm(self, messages: list, temperature: float = None) -> str:
        """Call the LLM and return the response text."""
        temp = temperature if temperature is not None else self.temperature
        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={"temperature": temp}
        )
        return response["message"]["content"]

    def log(self, message: str, prefix: str = ""):
        """Print a log message if verbose mode is on."""
        if self.verbose:
            print(f"{prefix}{message}")

    def log_step(self, step: AgentStep):
        """Print a formatted step summary."""
        if self.verbose:
            print(str(step))

    def get_step_log(self) -> list:
        """Return all steps as a list of dicts for saving or analysis."""
        return [s.to_dict() for s in self.steps]

    def save_log(self, filename: str):
        """Save the full step log to a JSON file."""
        import os
        os.makedirs("agent_logs", exist_ok=True)
        filepath = f"agent_logs/{filename}"
        with open(filepath, "w") as f:
            json.dump(self.get_step_log(), f, indent=2)
        print(f"\n[Log saved to {filepath}]")

    def reset(self):
        """Reset agent state for a new task."""
        self.steps = []
        self.start_time = None