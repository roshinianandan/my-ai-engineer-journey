import ollama
import argparse
from knowledge_graph.builder import KnowledgeGraphBuilder
from knowledge_graph.traversal import GraphTraversal
from config import MODEL


class GraphRAG:
    """
    GraphRAG — combines knowledge graph traversal with vector RAG.

    Standard RAG: query → similar chunks → answer
    GraphRAG:     query → similar chunks + graph context → better answer

    The graph context adds:
    - Entity relationships the chunks may not explicitly state
    - Multi-hop connections between concepts
    - Structured knowledge that complements unstructured text
    """

    def __init__(self):
        self.builder = KnowledgeGraphBuilder()
        self.traversal = GraphTraversal(self.builder)

    def build_graph_from_kb(self, top_k: int = 15):
        """Build the knowledge graph from the indexed knowledge base."""
        self.builder.build_from_knowledge_base(top_k=top_k)
        self.builder.print_stats()

    def answer(
        self,
        query: str,
        use_vector_rag: bool = True,
        use_graph: bool = True,
        top_k: int = 3
    ) -> dict:
        """
        Answer a question using GraphRAG.

        Combines:
        1. Vector RAG — semantic chunk retrieval
        2. Graph context — entity relationships from knowledge graph
        """
        print(f"\n{'='*55}")
        print(f"  GRAPH RAG QUERY")
        print(f"  Query: {query}")
        print(f"{'='*55}\n")

        vector_context = ""
        graph_context = ""
        sources = []

        # Step 1: Vector RAG retrieval
        if use_vector_rag:
            print("[GraphRAG] Retrieving vector context...")
            try:
                from rag.knowledge_base import search
                chunks = search(query=query, top_k=top_k)
                if chunks:
                    vector_context = "\n\n".join(
                        f"[Source: {c['source']}]\n{c['text']}"
                        for c in chunks
                    )
                    sources = [c["source"] for c in chunks]
                    print(f"[GraphRAG] Found {len(chunks)} relevant chunks")
            except Exception as e:
                print(f"[GraphRAG] Vector search error: {e}")

        # Step 2: Graph context
        if use_graph and self.builder.graph.number_of_nodes() > 0:
            print("[GraphRAG] Extracting graph context...")
            graph_context = self.traversal.get_context_for_query(query)
            if graph_context:
                print(f"[GraphRAG] Graph context: {len(graph_context)} chars")

        # Step 3: Build combined prompt
        context_parts = []

        if vector_context:
            context_parts.append(f"DOCUMENT CONTEXT:\n{vector_context}")

        if graph_context:
            context_parts.append(
                f"KNOWLEDGE GRAPH CONTEXT (entity relationships):\n{graph_context}"
            )

        if not context_parts:
            combined_context = "No context available."
        else:
            combined_context = "\n\n" + "="*40 + "\n\n".join(context_parts)

        prompt = f"""Answer the question using the provided context.
The context includes both document excerpts and knowledge graph relationships.
Use both to give a comprehensive, accurate answer.
If the answer is not in the context, say you don't have enough information.

CONTEXT:
{combined_context}

QUESTION: {query}

ANSWER:"""

        print("\n[GraphRAG] Generating answer...")
        print("🤖 Answer: ", end="", flush=True)

        full_reply = ""
        for chunk in ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.3}
        ):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_reply += token

        print()

        return {
            "query": query,
            "answer": full_reply,
            "sources": sources,
            "used_vector_rag": use_vector_rag and bool(vector_context),
            "used_graph": use_graph and bool(graph_context),
            "graph_nodes": self.builder.graph.number_of_nodes()
        }

    def find_relationship(self, entity1: str, entity2: str) -> dict:
        """Find and explain the relationship between two entities."""
        print(f"\n[GraphRAG] Finding relationship: '{entity1}' ↔ '{entity2}'")
        path = self.traversal.find_path(entity1, entity2)

        if not path.get("connected"):
            return {
                "entity1": entity1,
                "entity2": entity2,
                "answer": f"No direct connection found between '{entity1}' and '{entity2}' in the knowledge graph."
            }

        # Use LLM to explain the path
        prompt = f"""Explain the relationship between these two concepts based on this path:

{path['path_description']}

Give a clear, concise explanation of how {entity1} and {entity2} are connected.
Keep it to 2-3 sentences."""

        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.4}
        )

        explanation = response["message"]["content"]
        print(f"Path: {path['path_description']}")
        print(f"Explanation: {explanation}")

        return {
            "entity1": entity1,
            "entity2": entity2,
            "path": path["path_description"],
            "hops": path["path_length"],
            "explanation": explanation
        }

    def explore_entity(self, entity_name: str, hops: int = 2):
        """Explore all connections from an entity."""
        result = self.traversal.multi_hop_query(entity_name, hops=hops)

        if "error" in result:
            print(f"[GraphRAG] {result['error']}")
            return result

        print(f"\n🔍 Exploring: {entity_name}")
        print(f"   Total entities reachable in {hops} hops: {result['total_entities']}")

        for hop_num, entities in result["reachable"].items():
            if entities:
                print(f"\n  Hop {hop_num}:")
                for e in entities[:5]:  # show max 5 per hop
                    ent = e if hop_num == 0 else e.get("entity", {})
                    name = ent.get("name", "unknown")
                    rel = e.get("relation", "") if hop_num > 0 else ""
                    direction = e.get("direction", "") if hop_num > 0 else ""
                    print(f"    {'→' if direction == 'outgoing' else '←'} "
                          f"[{rel}] {name}")

        return result


def interactive_graph_rag():
    """Interactive GraphRAG session."""
    rag = GraphRAG()

    print(f"\n🔬 GraphRAG System")
    print(f"   Graph: {rag.builder.graph.number_of_nodes()} entities, "
          f"{rag.builder.graph.number_of_edges()} relationships")
    print("\n   Commands:")
    print("   'build' — build graph from knowledge base")
    print("   'stats' — show graph statistics")
    print("   'explore <entity>' — explore entity connections")
    print("   'relate <entity1> to <entity2>' — find relationship")
    print("   'visualize' — save graph visualization")
    print("   'quit' — exit\n")
    print("-" * 55)

    while True:
        try:
            user_input = input("\nQuery: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        if user_input.lower() == "build":
            print("Building graph from knowledge base...")
            rag.build_graph_from_kb(top_k=20)
            continue

        if user_input.lower() == "stats":
            rag.builder.print_stats()
            continue

        if user_input.lower() == "visualize":
            rag.builder.visualize()
            continue

        if user_input.lower().startswith("explore "):
            entity = user_input[8:].strip()
            rag.explore_entity(entity)
            continue

        if " to " in user_input.lower() and user_input.lower().startswith("relate "):
            parts = user_input[7:].split(" to ", 1)
            if len(parts) == 2:
                rag.find_relationship(parts[0].strip(), parts[1].strip())
                continue

        # Default: GraphRAG answer
        rag.answer(user_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG System")
    parser.add_argument("--build",    action="store_true",
                        help="Build graph from knowledge base")
    parser.add_argument("--query",    type=str, help="Ask a question")
    parser.add_argument("--relate",   nargs=2, metavar=("ENTITY1", "ENTITY2"),
                        help="Find relationship between two entities")
    parser.add_argument("--explore",  type=str, help="Explore entity connections")
    parser.add_argument("--visualize",action="store_true",
                        help="Visualize the knowledge graph")
    parser.add_argument("--chat",     action="store_true",
                        help="Interactive GraphRAG session")
    parser.add_argument("--text",     type=str,
                        help="Build graph from custom text")
    args = parser.parse_args()

    rag = GraphRAG()

    if args.build:
        rag.build_graph_from_kb(top_k=20)
        rag.builder.print_stats()

    elif args.text:
        rag.builder.build_from_text(args.text)
        rag.builder.save()
        rag.builder.print_stats()

    elif args.visualize:
        rag.builder.visualize()

    elif args.relate:
        rag.find_relationship(args.relate[0], args.relate[1])

    elif args.explore:
        rag.explore_entity(args.explore)

    elif args.query:
        rag.answer(args.query)

    elif args.chat:
        interactive_graph_rag()

    else:
        interactive_graph_rag()