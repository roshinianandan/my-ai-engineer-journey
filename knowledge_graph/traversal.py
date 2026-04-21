import networkx as nx
from knowledge_graph.builder import KnowledgeGraphBuilder


class GraphTraversal:
    """
    Traverses the knowledge graph to answer questions.

    Methods:
    - find_entity: look up a node by name
    - get_neighbors: find connected entities
    - find_path: shortest path between two entities
    - get_subgraph: extract a focused subgraph
    - multi_hop: follow relationship chains
    """

    def __init__(self, builder: KnowledgeGraphBuilder):
        self.graph = builder.graph

    def find_entity(self, name: str) -> dict | None:
        """Find an entity by name (fuzzy match)."""
        name_lower = name.lower().strip()

        # Exact match
        if name_lower in self.graph:
            return {
                "id": name_lower,
                **self.graph.nodes[name_lower]
            }

        # Partial match
        for node in self.graph.nodes:
            if name_lower in node or node in name_lower:
                return {
                    "id": node,
                    **self.graph.nodes[node]
                }

        return None

    def get_neighbors(
        self,
        entity_name: str,
        direction: str = "both",
        max_results: int = 10
    ) -> dict:
        """
        Get all entities connected to an entity.
        direction: 'out' (what this entity points to),
                   'in' (what points to this entity),
                   'both'
        """
        entity = self.find_entity(entity_name)
        if not entity:
            return {"error": f"Entity '{entity_name}' not found"}

        node_id = entity["id"]
        neighbors = {"entity": entity, "outgoing": [], "incoming": []}

        if direction in ["out", "both"]:
            for successor in list(self.graph.successors(node_id))[:max_results]:
                edge_data = self.graph.edges[node_id, successor]
                neighbors["outgoing"].append({
                    "entity": {
                        "id": successor,
                        **self.graph.nodes[successor]
                    },
                    "relation": edge_data.get("relation", "related to")
                })

        if direction in ["in", "both"]:
            for predecessor in list(self.graph.predecessors(node_id))[:max_results]:
                edge_data = self.graph.edges[predecessor, node_id]
                neighbors["incoming"].append({
                    "entity": {
                        "id": predecessor,
                        **self.graph.nodes[predecessor]
                    },
                    "relation": edge_data.get("relation", "related to")
                })

        return neighbors

    def find_path(
        self,
        source: str,
        target: str
    ) -> dict:
        """
        Find the shortest path between two entities.
        This answers "How is X connected to Y?" questions.
        """
        source_entity = self.find_entity(source)
        target_entity = self.find_entity(target)

        if not source_entity:
            return {"error": f"Entity '{source}' not found"}
        if not target_entity:
            return {"error": f"Entity '{target}' not found"}

        source_id = source_entity["id"]
        target_id = target_entity["id"]

        try:
            # Try directed path first
            path = nx.shortest_path(
                self.graph, source_id, target_id
            )
        except nx.NetworkXNoPath:
            try:
                # Try undirected path
                undirected = self.graph.to_undirected()
                path = nx.shortest_path(undirected, source_id, target_id)
            except nx.NetworkXNoPath:
                return {
                    "source": source,
                    "target": target,
                    "path": None,
                    "connected": False,
                    "message": f"No path found between '{source}' and '{target}'"
                }

        # Build path description with relations
        path_description = []
        for i in range(len(path) - 1):
            src_node = path[i]
            tgt_node = path[i + 1]
            src_name = self.graph.nodes[src_node].get("name", src_node)
            tgt_name = self.graph.nodes[tgt_node].get("name", tgt_node)

            # Get relation (try both directions)
            if self.graph.has_edge(src_node, tgt_node):
                relation = self.graph.edges[src_node, tgt_node].get("relation", "→")
            elif self.graph.has_edge(tgt_node, src_node):
                relation = self.graph.edges[tgt_node, src_node].get("relation", "←")
            else:
                relation = "→"

            path_description.append(
                f"{src_name} --[{relation}]--> {tgt_name}"
            )

        return {
            "source": source,
            "target": target,
            "path": path,
            "path_length": len(path) - 1,
            "path_description": " | ".join(path_description),
            "connected": True
        }

    def multi_hop_query(
        self,
        start_entity: str,
        hops: int = 2
    ) -> dict:
        """
        Explore the graph starting from an entity for N hops.
        Returns all entities reachable within N steps.
        This is how GraphRAG answers complex relational questions.
        """
        entity = self.find_entity(start_entity)
        if not entity:
            return {"error": f"Entity '{start_entity}' not found"}

        start_id = entity["id"]
        reachable = {0: [entity]}

        current_nodes = {start_id}
        visited = {start_id}

        for hop in range(1, hops + 1):
            next_nodes = set()
            hop_entities = []

            for node_id in current_nodes:
                # Get both successors and predecessors
                connected = (
                    set(self.graph.successors(node_id)) |
                    set(self.graph.predecessors(node_id))
                )

                for neighbor in connected:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_nodes.add(neighbor)

                        # Get relation
                        if self.graph.has_edge(node_id, neighbor):
                            rel = self.graph.edges[node_id, neighbor].get("relation", "→")
                            direction = "outgoing"
                        else:
                            rel = self.graph.edges[neighbor, node_id].get("relation", "←")
                            direction = "incoming"

                        hop_entities.append({
                            "entity": {
                                "id": neighbor,
                                **self.graph.nodes[neighbor]
                            },
                            "relation": rel,
                            "direction": direction,
                            "from": node_id
                        })

            reachable[hop] = hop_entities
            current_nodes = next_nodes

            if not next_nodes:
                break

        total = sum(len(v) for v in reachable.values())
        return {
            "start": start_entity,
            "hops": hops,
            "total_entities": total,
            "reachable": reachable
        }

    def get_context_for_query(self, query: str, max_entities: int = 5) -> str:
        """
        Extract relevant graph context for a natural language query.
        Used by GraphRAG to enrich prompts with graph knowledge.
        """
        words = query.lower().split()
        relevant_nodes = []

        for node in self.graph.nodes:
            node_name = self.graph.nodes[node].get("name", node).lower()
            if any(word in node_name or node_name in word
                   for word in words if len(word) > 3):
                relevant_nodes.append(node)

        if not relevant_nodes:
            return ""

        context_parts = []
        seen = set()

        for node_id in relevant_nodes[:max_entities]:
            # Add node info
            node_data = self.graph.nodes[node_id]
            name = node_data.get("name", node_id)
            desc = node_data.get("description", "")
            node_type = node_data.get("type", "")

            context_parts.append(f"Entity: {name} ({node_type})")
            if desc:
                context_parts.append(f"  Description: {desc}")

            # Add outgoing relations
            for successor in self.graph.successors(node_id):
                if successor not in seen:
                    seen.add(successor)
                    rel = self.graph.edges[node_id, successor].get("relation", "→")
                    target_name = self.graph.nodes[successor].get("name", successor)
                    context_parts.append(f"  → [{rel}] → {target_name}")

            # Add incoming relations
            for predecessor in self.graph.predecessors(node_id):
                if predecessor not in seen:
                    seen.add(predecessor)
                    rel = self.graph.edges[predecessor, node_id].get("relation", "←")
                    source_name = self.graph.nodes[predecessor].get("name", predecessor)
                    context_parts.append(f"  ← [{rel}] ← {source_name}")

            context_parts.append("")

        return "\n".join(context_parts)