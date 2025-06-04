from typing import List, Optional, Dict, Any
from models.knowledge_graph import KnowledgeGraph # Assuming KnowledgeGraph is defined
from models.document import Document # Example, adjust as needed
from models.super_token import SuperToken
import gmpy2 # If primes are used here

class KnowledgeGraphService:
    def __init__(self, knowledge_graph: Optional[KnowledgeGraph] = None):
        self.graph = knowledge_graph if knowledge_graph else KnowledgeGraph(name="MainKG")
        # TODO: Initialize prime assignment strategy related data if needed
        # e.g., self.prime_map_for_nodes = {}
        
        # Storage for SuperTokens
        self.super_tokens: Dict[str, SuperToken] = {} # Store by SuperToken ID

    def add_document_to_graph(self, document: Document):
        # TODO: Logic to add document and its chunks as nodes,
        # and 'contains' edges.
        print(f"Placeholder: Adding document {document.title} to graph.")
        pass

    def get_sorted_neighbor_labels(self, source_node_label: str) -> List[str]:
        """
        Returns a sorted list of labels of neighbors for a given source node.
        Needed for canonical prime assignment.
        """
        if not self.graph or not self.graph.graph or source_node_label not in self.graph.graph:
            return []
        try:
            neighbor_labels = sorted(list(self.graph.graph.neighbors(source_node_label)))
            return neighbor_labels
        except Exception as e:
            print(f"Error getting sorted neighbors for {source_node_label}: {e}")
            return []

    def get_canonical_edge_prime(self, source_node_label: str, target_node_label: str) -> Optional[gmpy2.mpz]:
        """
        Gets the canonical prime number for an edge based on a stable ordering.
        Strategy: Sort target node labels alphabetically/numerically from source.
        """
        from services.path_encoding_service import get_nth_prime_gmpy
        neighbor_labels = self.get_sorted_neighbor_labels(source_node_label)
        if not neighbor_labels:
            print(f"Warning: Node {source_node_label} has no neighbors or does not exist in graph.")
            return None
        if target_node_label not in neighbor_labels:
            print(f"Warning: Target {target_node_label} not a neighbor of {source_node_label}.")
            return None
        try:
            prime_order_index = neighbor_labels.index(target_node_label) + 1
            return get_nth_prime_gmpy(prime_order_index)
        except ValueError:
            print(f"Error: Target {target_node_label} was in neighbor_labels but index failed (should not happen).")
            return None
        except Exception as e:
            print(f"Error getting canonical edge prime for ({source_node_label} -> {target_node_label}): {e}")
            return None

    def find_path(self, start_node_label: str, end_node_label: str) -> Optional[List[str]]:
        """
        Finds a path (list of node labels) between two nodes using BFS (NetworkX shortest_path).
        """
        if not self.graph or not self.graph.graph:
            print("Error: Graph not initialized in KnowledgeGraphService.")
            return None
        if start_node_label not in self.graph.graph or end_node_label not in self.graph.graph:
            print(f"Error: Start ({start_node_label}) or End ({end_node_label}) node not in graph.")
            return None
        try:
            import networkx as nx
            path_nodes = nx.shortest_path(self.graph.graph, source=start_node_label, target=end_node_label)
            return path_nodes
        except nx.NetworkXNoPath:
            print(f"No path found between {start_node_label} and {end_node_label}.")
            return None
        except Exception as e:
            print(f"Error finding path between {start_node_label} and {end_node_label}: {e}")
            return None

    def register_super_token(self,
                             path_node_labels: List[str],
                             edge_atomic_primes: List[Any],
                             encoded_c: Any,
                             encoded_d: int,
                             claim: str) -> Optional[SuperToken]:
        """
        Creates a SuperToken from a path and its encoding, and registers it.
        """
        if not path_node_labels or len(path_node_labels) < 2:
            print("Error: Path for SuperToken must have at least 2 nodes (1 edge).")
            return None
        if not claim:
            print("Error: SuperToken must have a claim.")
            return None
        if not edge_atomic_primes or len(edge_atomic_primes) != len(path_node_labels) -1:
            print("Error: Mismatch between path length and number of edge primes for SuperToken.")
            return None

        st_instance = SuperToken(
            claim=claim,
            path_node_labels=list(path_node_labels), # Store a copy
            edge_atomic_primes=list(edge_atomic_primes), # Store a copy
            path_code_c=encoded_c,
            path_code_d=encoded_d
        )
        self.super_tokens[st_instance.id] = st_instance
        print(f"Registered SuperToken: {st_instance}")

        # Optional: Add SuperToken as a node in the main KnowledgeGraph
        # This requires defining how a SuperToken node itself is represented (properties, etc.)
        # For now, we just store it in the service's dictionary.
        # Example:
        # st_node = KnowledgeNode(id=st_instance.id, node_type='super_token', content_id=st_instance.id, properties={'claim': claim})
        # self.graph.add_node(st_node) # self.graph is the KnowledgeGraph model instance
        
        return st_instance

    def get_node_degree(self, node_label: str) -> int:
        """
        Returns the degree (number of connections) of the given node label in the graph.
        """
        if not self.graph or not self.graph.graph:
            return 0
        if node_label not in self.graph.graph:
            return 0
        try:
            return self.graph.graph.degree[node_label]
        except Exception as e:
            print(f"Error getting degree for node {node_label}: {e}")
            return 0 