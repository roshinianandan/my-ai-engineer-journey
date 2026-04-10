import ollama
from config import MODEL


class ShortTermMemory:
    """
    Manages the in-session conversation buffer.
    Keeps the last N messages to avoid overflowing the context window.
    When the buffer is full, older messages are summarized and compressed.
    """

    def __init__(self, max_messages: int = 10, system_prompt: str = ""):
        self.max_messages = max_messages
        self.system_prompt = system_prompt
        self.messages = []
        self.summary = ""   # compressed summary of older messages

    def add(self, role: str, content: str):
        """Add a message to the buffer."""
        self.messages.append({"role": role, "content": content})

        # If buffer is full compress older messages into a summary
        if len(self.messages) > self.max_messages:
            self._compress()

    def _compress(self):
        """
        Summarize the oldest half of messages to free up buffer space.
        This is how chatbots handle very long conversations without
        running out of context window space.
        """
        half = len(self.messages) // 2
        older = self.messages[:half]
        self.messages = self.messages[half:]

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in older
        )

        summary_prompt = f"""Summarize this conversation excerpt in 3 sentences.
Capture the key topics, decisions, and any important facts the user shared.

Conversation:
{conversation_text}

Summary:"""

        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            stream=False,
            options={"temperature": 0.3}
        )
        new_summary = response["message"]["content"]

        if self.summary:
            self.summary += " " + new_summary
        else:
            self.summary = new_summary

        print(f"\n[Memory compressed — {half} messages summarized]\n")

    def get_messages(self) -> list:
        """
        Return the full message list for sending to the LLM.
        Includes system prompt and summary of older messages if available.
        """
        full = []

        # Build system message with summary context
        system_content = self.system_prompt
        if self.summary:
            system_content += (
                f"\n\nEarlier in this conversation (summary):\n{self.summary}"
            )

        if system_content:
            full.append({"role": "system", "content": system_content})

        full.extend(self.messages)
        return full

    def clear(self):
        """Reset all memory."""
        self.messages = []
        self.summary = ""

    def stats(self) -> dict:
        return {
            "messages_in_buffer": len(self.messages),
            "has_summary": bool(self.summary),
            "summary_length": len(self.summary)
        }