import json
import os
import networkx as nx
from knowledge_graph.entity_extractor import extract_entities_and_relations


GRAPH_PATH = "knowledge_graph/graph_data.json"


class KnowledgeGraphBuilder:
    """
    Builds and manages a knowledge graph of entities and relationships.

    Uses NetworkX as the underlying graph structure.
    Nodes = entities, Edges = relationships.

    The graph can be:
    - Built from text documents
    - Saved to and loaded from disk
    - Queried for entities and relationships
    - Visualized
    """

    def __init__(self, graph_path: str = GRAPH_PATH):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()  # Directed graph
        self._load_if_exists()

    def _load_if_exists(self):
        """Load existing graph from disk if available."""
        if os.path.exists(self.graph_path):
            self.load()
            print(f"[KG] Loaded existing graph: "
                  f"{self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")
        else:
            print("[KG] Starting with empty graph")

    def add_entity(
        self,
        name: str,
        entity_type: str = "CONCEPT",
        description: str = "",
        sources: list = None
    ):
        """Add an entity as a node in the graph."""
        node_id = name.lower().strip()
        self.graph.add_node(
            node_id,
            name=name,
            type=entity_type,
            description=description,
            sources=sources or []
        )

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        doc_source: str = ""
    ):
        """Add a relationship as a directed edge."""
        source_id = source.lower().strip()
        target_id = target.lower().strip()

        # Auto-create nodes if they don't exist
        if source_id not in self.graph:
            self.add_entity(source)
        if target_id not in self.graph:
            self.add_entity(target)

        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation,
            doc_source=doc_source
        )

    def build_from_text(self, text: str, source: str = "manual") -> dict:
        """Extract entities and relations from text and add to graph."""
        print(f"[KG] Extracting from text ({len(text)} chars)...")
        result = extract_entities_and_relations(text)

        for entity in result.get("entities", []):
            self.add_entity(
                name=entity["name"],
                entity_type=entity.get("type", "CONCEPT"),
                description=entity.get("description", ""),
                sources=[source]
            )

        for rel in result.get("relations", []):
            self.add_relation(
                source=rel["source"],
                relation=rel["relation"],
                target=rel["target"],
                doc_source=source
            )

        print(f"[KG] Graph now has {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return result

    def build_from_documents(self, documents: list):
        """Build graph from a list of document dicts."""
        for doc in documents:
            self.build_from_text(
                text=doc.get("text", ""),
                source=doc.get("source", "unknown")
            )
        self.save()

    def build_from_knowledge_base(self, top_k: int = 20):
        """Pull documents from ChromaDB and build graph from them."""
        from vector_store import get_or_create_collection
        collection = get_or_create_collection("knowledge_base")

        if collection.count() == 0:
            print("[KG] Knowledge base is empty. Index documents first.")
            return

        print(f"[KG] Building graph from knowledge base "
              f"({collection.count()} chunks)...")

        all_items = collection.get(
            include=["documents", "metadatas"],
            limit=top_k
        )

        documents = []
        for doc, meta in zip(
            all_items["documents"],
            all_items["metadatas"]
        ):
            documents.append({
                "text": doc,
                "source": meta.get("source", "unknown")
            })

        self.build_from_documents(documents)

    def save(self):
        """Save graph to disk as JSON."""
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)

        data = {
            "nodes": [
                {
                    "id": node,
                    **self.graph.nodes[node]
                }
                for node in self.graph.nodes
            ],
            "edges": [
                {
                    "source": src,
                    "target": tgt,
                    **self.graph.edges[src, tgt]
                }
                for src, tgt in self.graph.edges
            ]
        }

        with open(self.graph_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[KG] Graph saved: {self.graph_path}")

    def load(self):
        """Load graph from disk."""
        with open(self.graph_path) as f:
            data = json.load(f)

        self.graph.clear()

        for node in data.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)

        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            self.graph.add_edge(src, tgt, **edge)

    def stats(self) -> dict:
        """Return graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "entity_types": {}, "density": 0.0}

        entity_types = {}
        for node in self.graph.nodes:
            t = self.graph.nodes[node].get("type", "UNKNOWN")
            entity_types[t] = entity_types.get(t, 0) + 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entity_types": entity_types,
            "density": round(nx.density(self.graph), 4)
        }

    def print_stats(self):
        """Print graph statistics."""
        s = self.stats()
        print(f"\n{'='*50}")
        print(f"  KNOWLEDGE GRAPH STATS")
        print(f"{'='*50}")
        print(f"  Nodes (entities): {s['nodes']}")
        print(f"  Edges (relations): {s['edges']}")
        print(f"  Density: {s['density']}")
        print(f"  Entity types: {s['entity_types']}")
        print(f"{'='*50}\n")

    def visualize(self, max_nodes: int = 30, save_path: str = None):
        """Visualize the knowledge graph using matplotlib."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend
            import matplotlib.pyplot as plt

            subgraph_nodes = list(self.graph.nodes)[:max_nodes]
            subgraph = self.graph.subgraph(subgraph_nodes)

            plt.figure(figsize=(16, 12))
            pos = nx.spring_layout(subgraph, k=2, seed=42)

            # Color nodes by type
            type_colors = {
                "CONCEPT": "#4E79A7",
                "TECHNOLOGY": "#F28E2B",
                "PERSON": "#E15759",
                "ORGANIZATION": "#76B7B2",
                "PROCESS": "#59A14F"
            }

            for entity_type, color in type_colors.items():
                nodes_of_type = [
                    n for n in subgraph.nodes
                    if subgraph.nodes[n].get("type") == entity_type
                ]
                if nodes_of_type:
                    nx.draw_networkx_nodes(
                        subgraph, pos,
                        nodelist=nodes_of_type,
                        node_color=color,
                        node_size=1500,
                        alpha=0.9,
                        label=entity_type
                    )

            # Draw labels
            labels = {
                n: subgraph.nodes[n].get("name", n)[:20]
                for n in subgraph.nodes
            }
            nx.draw_networkx_labels(
                subgraph, pos, labels,
                font_size=8, font_weight="bold"
            )

            # Draw edges
            nx.draw_networkx_edges(
                subgraph, pos,
                edge_color="#999999",
                arrows=True,
                arrowsize=15,
                width=1.5,
                alpha=0.6
            )

            # Edge labels
            edge_labels = {
                (src, tgt): subgraph.edges[src, tgt].get("relation", "")[:15]
                for src, tgt in subgraph.edges
            }
            nx.draw_networkx_edge_labels(
                subgraph, pos, edge_labels,
                font_size=6,
                alpha=0.8
            )

            plt.title(
                f"Knowledge Graph — {subgraph.number_of_nodes()} entities, "
                f"{subgraph.number_of_edges()} relationships",
                fontsize=14
            )
            plt.legend(loc="upper left", fontsize=9)
            plt.axis("off")
            plt.tight_layout()

            save_file = save_path or "knowledge_graph/graph_visualization.png"
            plt.savefig(save_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[KG] Graph visualization saved: {save_file}")

        except Exception as e:
            print(f"[KG] Visualization error: {e}")