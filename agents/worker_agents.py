import ollama
from agents.message_bus import MessageBus, MessageType, Priority
from config import MODEL


class BaseWorkerAgent:
    """Base class for all worker agents."""

    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id = agent_id
        self.bus = bus
        self.bus.register_agent(agent_id)
        self.completed_tasks = []

    def call_llm(self, prompt: str, temperature: float = 0.5) -> str:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": temperature}
        )
        return response["message"]["content"]

    def report_result(
        self,
        result: str,
        task_id: str,
        recipient: str = "orchestrator"
    ):
        """Send task result back to orchestrator."""
        self.bus.send(
            sender=self.agent_id,
            recipient=recipient,
            message_type=MessageType.RESULT,
            content={"result": result, "task_id": task_id},
            parent_id=task_id
        )

    def report_error(self, error: str, task_id: str):
        """Report a task failure."""
        self.bus.send(
            sender=self.agent_id,
            recipient="orchestrator",
            message_type=MessageType.ERROR,
            content={"error": error, "task_id": task_id},
            priority=Priority.HIGH
        )


class SearcherAgent(BaseWorkerAgent):
    """
    Specializes in searching the knowledge base for relevant information.
    Takes a search query and returns the most relevant text chunks.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("searcher", bus)

    def execute(self, task: dict) -> str:
        """Search the knowledge base for information."""
        query = task.get("query", "")
        top_k = task.get("top_k", 3)

        print(f"\n[Searcher] Searching for: '{query}'")

        try:
            from rag.knowledge_base import search
            chunks = search(query=query, top_k=top_k)

            if not chunks:
                return f"No relevant information found for: {query}"

            results = []
            for i, chunk in enumerate(chunks, 1):
                results.append(
                    f"[{i}] Source: {chunk['source']} | "
                    f"Score: {chunk['score']}\n{chunk['text']}"
                )

            return "\n\n".join(results)

        except Exception as e:
            return f"Search failed: {str(e)}"


class SummarizerAgent(BaseWorkerAgent):
    """
    Specializes in summarizing long text into concise summaries.
    Takes raw text and returns a structured summary.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("summarizer", bus)

    def execute(self, task: dict) -> str:
        """Summarize provided text."""
        text = task.get("text", "")
        style = task.get("style", "concise")
        max_sentences = task.get("max_sentences", 3)

        print(f"\n[Summarizer] Summarizing {len(text)} chars in {style} style...")

        prompt = f"""Summarize the following text in {max_sentences} clear sentences.
Style: {style}
Focus on the most important facts and insights.

Text:
{text[:3000]}

Summary:"""

        return self.call_llm(prompt, temperature=0.3)


class CriticAgent(BaseWorkerAgent):
    """
    Specializes in evaluating and critiquing content quality.
    Reviews summaries and findings for accuracy and completeness.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("critic", bus)

    def execute(self, task: dict) -> str:
        """Critique a piece of content."""
        content = task.get("content", "")
        original_query = task.get("original_query", "")
        criteria = task.get("criteria", [
            "accuracy", "completeness", "clarity"
        ])

        print(f"\n[Critic] Evaluating content quality...")

        prompt = f"""Evaluate this content based on the original question.

Original Question: {original_query}

Content to Evaluate:
{content}

Evaluate on these criteria: {', '.join(criteria)}

For each criterion:
- Score it 1-10
- Give one sentence of feedback

Then give an OVERALL score (1-10) and one key improvement suggestion.

Format:
Accuracy: [score] - [feedback]
Completeness: [score] - [feedback]
Clarity: [score] - [feedback]
Overall: [score] - [improvement suggestion]"""

        return self.call_llm(prompt, temperature=0.2)


class WriterAgent(BaseWorkerAgent):
    """
    Specializes in writing structured, polished reports.
    Takes raw findings and formats them into professional output.
    """

    def __init__(self, bus: MessageBus):
        super().__init__("writer", bus)

    def execute(self, task: dict) -> str:
        """Write a structured report from collected findings."""
        topic = task.get("topic", "")
        findings = task.get("findings", [])
        report_type = task.get("report_type", "research_summary")

        print(f"\n[Writer] Writing {report_type} on: '{topic}'")

        findings_text = "\n\n".join(
            f"Finding {i+1}:\n{f}"
            for i, f in enumerate(findings)
        )

        prompt = f"""Write a professional {report_type} on the topic: "{topic}"

Use the following findings as your source material:

{findings_text}

Structure your report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Analysis (2-3 paragraphs)
4. Conclusion (1-2 sentences)

Write clearly, professionally, and concisely."""

        return self.call_llm(prompt, temperature=0.4)