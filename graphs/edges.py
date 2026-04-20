from graphs.state import ResearchState


def should_revise(state: ResearchState) -> str:
    """
    Conditional edge: decide whether to revise or finalize.

    This is the key decision point in the graph.
    Returns the name of the next node to execute.

    LangGraph uses the return value to route to the correct node.
    """
    revision_needed = state.get("revision_needed", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    quality_score = state.get("quality_score", 10)

    print(f"\n[Edge: should_revise] "
          f"revision_needed={revision_needed} | "
          f"iteration={iteration} | "
          f"score={quality_score}")

    if revision_needed and iteration <= max_iterations:
        print(f"[Edge] → Routing to SUMMARIZE for revision {iteration}")
        return "revise"
    else:
        if quality_score >= 7:
            print(f"[Edge] → Routing to FINALIZE (quality passed)")
        else:
            print(f"[Edge] → Routing to FINALIZE (max iterations reached)")
        return "finalize"


def route_after_search(state: ResearchState) -> str:
    """
    Edge after search: always go to summarize.
    Could be extended to handle search failures differently.
    """
    sources = state.get("sources_found", 0)

    if sources == 0:
        print("[Edge] → No sources found, going to summarize with LLM knowledge")
    else:
        print(f"[Edge] → Found {sources} sources, going to summarize")

    return "summarize"