from langchain_core.messages import HumanMessage, AIMessage


class AgentMemoryManager:
    """
    Manages conversation memory for LangChain agents.
    Uses langchain_core only — no deprecated langchain.memory dependency.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._messages: list = []  # stores HumanMessage / AIMessage objects

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trimmed(self) -> list:
        """Return only the last `window_size` pairs (2 * window_size msgs)."""
        max_msgs = self.window_size * 2
        return self._messages[-max_msgs:] if len(self._messages) > max_msgs else self._messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_interaction(self, human_input: str, ai_output: str):
        """Add one human/AI exchange to memory."""
        self._messages.append(HumanMessage(content=human_input))
        self._messages.append(AIMessage(content=ai_output))

    def get_history(self) -> list:
        """Return windowed history as plain dicts."""
        history = []
        for msg in self._trimmed():
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "ai", "content": msg.content})
        return history

    def get_messages(self) -> list:
        """Return raw LangChain message objects (windowed)."""
        return self._trimmed()

    def show_history(self):
        """Pretty-print the current conversation window."""
        history = self.get_history()
        if not history:
            print("[No conversation history yet]")
            return
        print(f"\n📋 Conversation History ({len(history)} messages):\n")
        for msg in history:
            role = "You" if msg["role"] == "human" else "Agent"
            content = msg["content"]
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"  {role}: {content}\n")

    def clear(self):
        """Wipe all stored messages."""
        self._messages.clear()
        print("[Agent memory cleared]")

    def stats(self) -> dict:
        """Return stats about current memory state."""
        history = self.get_history()
        return {
            "total_messages": len(history),
            "window_size": self.window_size,
            "human_messages": sum(1 for m in history if m["role"] == "human"),
            "ai_messages": sum(1 for m in history if m["role"] == "ai"),
        }