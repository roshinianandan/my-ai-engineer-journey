import re
import ollama
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from config import MODEL


# Facts worth remembering — patterns to detect in conversation
MEMORY_TRIGGERS = [
    r"my name is (.+)",
    r"i am (.+) years old",
    r"i work (?:at|for|as) (.+)",
    r"i live in (.+)",
    r"i (?:study|am studying) (.+)",
    r"i (?:like|love|enjoy|prefer) (.+)",
    r"i (?:don't|do not|dislike|hate) (.+)",
    r"my (?:goal|aim|target) is (.+)",
    r"i am (?:a|an) (.+)",
    r"my (?:favourite|favorite) (.+) is (.+)",
]


class MemoryManager:
    """
    Orchestrates short-term and long-term memory together.
    Decides what to save, when to recall, and how to inject
    memories into the prompt before each LLM call.
    """

    def __init__(
        self,
        user_id: str = "default",
        system_prompt: str = "",
        max_buffer: int = 10
    ):
        self.user_id = user_id
        self.short_term = ShortTermMemory(
            max_messages=max_buffer,
            system_prompt=system_prompt
        )
        self.long_term = LongTermMemory(user_id=user_id)
        self.system_prompt = system_prompt

    def process_input(self, user_input: str):
        """
        Before sending to LLM:
        1. Check if input contains facts worth saving
        2. Retrieve relevant long-term memories
        3. Build the enriched prompt with memories injected
        """
        # Auto-detect and save memorable facts
        self._auto_extract_facts(user_input)

        # Retrieve relevant memories
        memories = self.long_term.recall(user_input, top_k=3)

        # Add user message to short-term buffer
        self.short_term.add("user", user_input)

        # Build message list with memories injected
        messages = self._build_messages_with_memory(memories)

        return messages

    def process_response(self, assistant_response: str):
        """Save assistant response to short-term buffer."""
        self.short_term.add("assistant", assistant_response)

    def _auto_extract_facts(self, text: str):
        """
        Use regex patterns to detect facts the user is sharing about themselves.
        Automatically saves them to long-term memory.
        """
        text_lower = text.lower()
        for pattern in MEMORY_TRIGGERS:
            match = re.search(pattern, text_lower)
            if match:
                fact = text.strip()
                self.long_term.save(fact, category="personal")
                break

    def _build_messages_with_memory(self, memories: list) -> list:
        """
        Inject relevant long-term memories into the system prompt
        so the LLM can use them when generating a response.
        """
        messages = self.short_term.get_messages()

        if not memories:
            return messages

        memory_text = "\n".join(
            f"- {m['fact']} (relevance: {m['score']})"
            for m in memories
        )

        memory_injection = (
            f"\n\nRelevant things you remember about this user:\n{memory_text}"
            f"\nUse these naturally in your response if relevant."
        )

        # Inject into system message
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += memory_injection
        else:
            messages.insert(0, {
                "role": "system",
                "content": self.system_prompt + memory_injection
            })

        return messages

    def save_memory(self, fact: str, category: str = "general"):
        """Manually save a fact to long-term memory."""
        self.long_term.save(fact, category)

    def forget(self, keyword: str):
        """Delete memories containing a keyword."""
        return self.long_term.forget(keyword)

    def show_memories(self):
        """Display all stored long-term memories."""
        memories = self.long_term.list_all()
        if not memories:
            print("\n[No long-term memories stored yet]\n")
            return

        print(f"\n📋 Long-term memories ({len(memories)} total):\n")
        for m in memories:
            print(f"  [{m['category']}] {m['fact']}")
            print(f"           Saved: {m['timestamp'][:19]}\n")

    def stats(self):
        """Show memory statistics."""
        st = self.short_term.stats()
        lt = self.long_term.stats()
        print(f"\n📊 Memory Stats:")
        print(f"  Short-term buffer:  {st['messages_in_buffer']} messages")
        print(f"  Has summary:        {st['has_summary']}")
        print(f"  Long-term memories: {lt['total_memories']}")
        print(f"  Categories:         {lt['categories']}\n")