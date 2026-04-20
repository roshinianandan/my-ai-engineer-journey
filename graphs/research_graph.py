import time
import argparse
from langgraph.graph import StateGraph, END
from graphs.state import ResearchState
from graphs.nodes import (
    search_node,
    summarize_node,
    write_report_node,
    critic_node,
    finalize_node
)
from graphs.edges import should_revise, route_after_search


def build_research_graph():
    """
    Build the LangGraph research workflow.

    Graph structure:
    START → search → summarize → write_report → critic
                                      ↑               |
                                      |    revise      |
                                      └───────────────┘
                                                       |
                                                finalize → END

    The critic decides: pass quality threshold → finalize
                        below threshold → revise (loop back)
    """
    # Create the state graph
    workflow = StateGraph(ResearchState)

    # Add all nodes
    workflow.add_node("search", search_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("write_report", write_report_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("search")

    # Add edges (connections between nodes)
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "write_report")
    workflow.add_edge("write_report", "critic")

    # Conditional edge from critic
    # should_revise() returns "revise" or "finalize"
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "summarize",    # loop back for revision
            "finalize": "finalize"    # move to completion
        }
    )

    # Finalize goes to END
    workflow.add_edge("finalize", END)

    # Compile the graph
    return workflow.compile()


def run_research(
    topic: str,
    max_iterations: int = 2,
    verbose: bool = True
) -> dict:
    """
    Run the full LangGraph research workflow.
    """
    print(f"\n{'='*60}")
    print(f"  LANGGRAPH RESEARCH WORKFLOW")
    print(f"  Topic: {topic}")
    print(f"  Max revisions: {max_iterations}")
    print(f"{'='*60}")

    graph = build_research_graph()

    # Initial state
    initial_state = ResearchState(
        topic=topic,
        original_query=topic,
        search_results="",
        sources_found=0,
        summary="",
        report="",
        critique="",
        quality_score=0,
        revision_needed=False,
        messages=[f"Starting research on: {topic}"],
        iteration=0,
        max_iterations=max_iterations,
        status="starting"
    )

    start_time = time.time()

    # Run the graph
    final_state = graph.invoke(initial_state)

    elapsed = round(time.time() - start_time, 2)

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"\n{final_state['report']}")
    print(f"\n{'='*60}")
    print(f"  Quality Score:  {final_state['quality_score']}/10")
    print(f"  Revisions Made: {final_state['iteration'] - 1}")
    print(f"  Sources Used:   {final_state['sources_found']}")
    print(f"  Time:           {elapsed}s")
    print(f"{'='*60}")

    if verbose:
        print(f"\n📋 Workflow Log:")
        for msg in final_state.get("messages", []):
            print(f"   • {msg}")

    return {
        "topic": topic,
        "report": final_state["report"],
        "quality_score": final_state["quality_score"],
        "revisions": final_state["iteration"] - 1,
        "sources": final_state["sources_found"],
        "time_seconds": elapsed,
        "messages": final_state.get("messages", [])
    }


def visualize_graph():
    """Print a text representation of the graph structure."""
    print(f"\n{'='*60}")
    print("  LANGGRAPH WORKFLOW STRUCTURE")
    print(f"{'='*60}")
    print("""
  START
    │
    ▼
  [search] ──────────── Searches knowledge base
    │
    ▼
  [summarize] ◄──────── Also receives revision feedback
    │                   from critic when score < 7
    ▼
  [write_report] ─────── Writes structured report
    │
    ▼
  [critic] ─────────── Scores quality 1-10
    │
    ├── score >= 7 ──► [finalize] ──► END
    │
    └── score < 7  ──► [summarize] (revision loop)
                        (max 2 revisions)
""")


def interactive_graph():
    """Interactive LangGraph research session."""
    print(f"\n🔬 LangGraph Research System")
    print("   Each research task runs through the full graph.")
    print("   Low quality reports trigger automatic revision cycles.")
    print("   Type 'graph' to see the workflow structure.")
    print("   Type 'quit' to exit.\n")
    print("-" * 60)

    while True:
        try:
            topic = input("\nResearch Topic: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not topic:
            continue
        if topic.lower() == "quit":
            break
        if topic.lower() == "graph":
            visualize_graph()
            continue

        result = run_research(topic, max_iterations=2)
        print(f"\n[Done — Score: {result['quality_score']}/10 | "
              f"Revisions: {result['revisions']} | "
              f"Time: {result['time_seconds']}s]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Research Workflow")
    parser.add_argument("--topic",   type=str, help="Research topic")
    parser.add_argument("--chat",    action="store_true", help="Interactive mode")
    parser.add_argument("--graph",   action="store_true", help="Show graph structure")
    parser.add_argument("--revisions", type=int, default=2,
                        help="Max revision cycles")
    args = parser.parse_args()

    if args.graph:
        visualize_graph()
    elif args.topic:
        run_research(args.topic, max_iterations=args.revisions)
    elif args.chat:
        interactive_graph()
    else:
        interactive_graph()