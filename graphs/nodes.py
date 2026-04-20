import ollama
from graphs.state import ResearchState
from config import MODEL


def call_llm(prompt: str, temperature: float = 0.4) -> str:
    """Helper to call Ollama."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": temperature}
    )
    return response["message"]["content"]


def search_node(state: ResearchState) -> dict:
    """
    Node 1: Search the knowledge base.
    Reads: topic
    Writes: search_results, sources_found, messages
    """
    topic = state["topic"]
    print(f"\n[Node: Search] Searching for: '{topic}'")

    try:
        from rag.knowledge_base import search
        chunks = search(query=topic, top_k=4)

        if not chunks:
            search_text = f"No specific information found about '{topic}' in the knowledge base."
            sources = 0
        else:
            parts = []
            for i, c in enumerate(chunks, 1):
                parts.append(
                    f"[Source {i}: {c['source']} | Score: {c['score']}]\n{c['text']}"
                )
            search_text = "\n\n---\n\n".join(parts)
            sources = len(chunks)

    except Exception as e:
        search_text = f"Search error: {str(e)}. Using LLM knowledge only."
        sources = 0

    print(f"[Node: Search] Found {sources} relevant chunks")

    return {
        "search_results": search_text,
        "sources_found": sources,
        "status": "searched",
        "messages": [f"Search completed: {sources} sources found"]
    }


def summarize_node(state: ResearchState) -> dict:
    """
    Node 2: Summarize search results.
    Reads: topic, search_results
    Writes: summary, messages
    """
    topic = state["topic"]
    search_results = state.get("search_results", "")
    iteration = state.get("iteration", 0)

    print(f"\n[Node: Summarize] Summarizing findings (iteration {iteration})")

    critique = state.get("critique", "")
    improvement_note = ""
    if critique and iteration > 0:
        improvement_note = f"\n\nPrevious critique to address:\n{critique}\nPlease improve based on this feedback."

    prompt = f"""Based on the following search results, create a comprehensive summary about: "{topic}"

Search Results:
{search_results[:3000]}

Write a clear, accurate summary covering:
1. Main concepts and definitions
2. Key insights and important details
3. Practical applications or examples
4. Any limitations or considerations
{improvement_note}

Summary:"""

    summary = call_llm(prompt, temperature=0.3)
    print(f"[Node: Summarize] Summary created: {len(summary)} chars")

    return {
        "summary": summary,
        "status": "summarized",
        "messages": [f"Summary created ({len(summary)} chars)"]
    }


def write_report_node(state: ResearchState) -> dict:
    """
    Node 3: Write a structured research report.
    Reads: topic, summary, critique (if revision)
    Writes: report, messages
    """
    topic = state["topic"]
    summary = state.get("summary", "")
    iteration = state.get("iteration", 0)
    critique = state.get("critique", "")

    print(f"\n[Node: Writer] Writing report (iteration {iteration})")

    revision_note = ""
    if critique and iteration > 0:
        revision_note = f"\n\nIMPORTANT: This is revision {iteration}. Address this feedback:\n{critique}"

    prompt = f"""Write a professional research report on: "{topic}"

Use this summary as your source material:
{summary}
{revision_note}

Structure the report with:
## Executive Summary
(2-3 sentences capturing the essence)

## Key Findings
(4-6 bullet points of the most important facts)

## Detailed Analysis
(2-3 paragraphs with depth and context)

## Conclusion
(1-2 sentences with the main takeaway)

Write professionally, clearly, and with appropriate technical depth."""

    report = call_llm(prompt, temperature=0.4)
    print(f"[Node: Writer] Report written: {len(report)} chars")

    return {
        "report": report,
        "status": "report_written",
        "messages": [f"Report written (iteration {iteration}, {len(report)} chars)"]
    }


def critic_node(state: ResearchState) -> dict:
    """
    Node 4: Critique the report quality.
    Reads: topic, report, original_query
    Writes: critique, quality_score, revision_needed, messages

    This is the key node that enables revision cycles —
    if quality is below threshold, the graph loops back.
    """
    topic = state["topic"]
    report = state.get("report", "")
    original_query = state.get("original_query", topic)
    iteration = state.get("iteration", 0)

    print(f"\n[Node: Critic] Evaluating report quality...")

    prompt = f"""You are a strict quality reviewer. Evaluate this research report.

Original Question: {original_query}
Topic: {topic}

Report to evaluate:
{report}

Evaluate on:
1. Accuracy — Is the information correct?
2. Completeness — Does it cover the topic well?
3. Clarity — Is it easy to understand?
4. Structure — Is it well organized?

Respond in EXACTLY this format:
SCORE: [number 1-10]
ACCURACY: [score 1-10] - [one sentence]
COMPLETENESS: [score 1-10] - [one sentence]
CLARITY: [score 1-10] - [one sentence]
STRUCTURE: [score 1-10] - [one sentence]
MAIN_ISSUE: [the single most important thing to improve]
VERDICT: [PASS if score >= 7, REVISE if score < 7]"""

    critique_response = call_llm(prompt, temperature=0.1)

    # Parse score
    import re
    score = 7  # default
    score_match = re.search(r"SCORE:\s*(\d+)", critique_response)
    if score_match:
        score = int(score_match.group(1))

    # Parse verdict
    revision_needed = "REVISE" in critique_response.upper()

    # Extract main issue
    issue_match = re.search(r"MAIN_ISSUE:\s*(.+?)(?=VERDICT:|$)", critique_response, re.DOTALL)
    main_issue = issue_match.group(1).strip() if issue_match else "Improve overall quality"

    print(f"[Node: Critic] Score: {score}/10 | Revision needed: {revision_needed}")
    print(f"[Node: Critic] Main issue: {main_issue}")

    # Increment iteration count
    new_iteration = iteration + 1

    return {
        "critique": critique_response,
        "quality_score": score,
        "revision_needed": revision_needed and new_iteration <= state.get("max_iterations", 2),
        "iteration": new_iteration,
        "status": "critiqued",
        "messages": [
            f"Critique: score={score}/10, "
            f"revision={'needed' if revision_needed else 'not needed'}"
        ]
    }


def finalize_node(state: ResearchState) -> dict:
    """
    Node 5: Finalize and format the output.
    Reads: report, quality_score, iteration, messages
    Writes: status, messages
    """
    quality_score = state.get("quality_score", 0)
    iteration = state.get("iteration", 0)
    report = state.get("report", "")

    print(f"\n[Node: Finalize] Finalizing report")
    print(f"[Node: Finalize] Final quality score: {quality_score}/10")
    print(f"[Node: Finalize] Revisions made: {iteration - 1}")

    return {
        "status": "complete",
        "messages": [
            f"Workflow complete | Score: {quality_score}/10 | "
            f"Revisions: {iteration - 1}"
        ]
    }