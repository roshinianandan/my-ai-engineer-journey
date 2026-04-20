from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    """
    Shared state passed between all nodes in the graph.
    Every node reads from this and writes to it.
    TypedDict ensures type safety across the workflow.

    Annotated[list, operator.add] means lists are
    appended rather than replaced when nodes update them.
    This is how LangGraph handles list accumulation.
    """
    # Input
    topic: str                              # The research topic
    original_query: str                     # Original user question

    # Search results
    search_results: str                     # Raw chunks from knowledge base
    sources_found: int                      # Number of sources retrieved

    # Processing
    summary: str                            # Summarized findings
    report: str                             # Full written report

    # Quality control
    critique: str                           # Critic's feedback
    quality_score: int                      # Score 1-10
    revision_needed: bool                   # Flag to trigger revision

    # Metadata
    messages: Annotated[list, operator.add] # Full message log
    iteration: int                          # Revision loop count
    max_iterations: int                     # Maximum revisions allowed
    status: str                             # Current workflow status