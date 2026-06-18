import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

class CommunityDetector:
    def __init__(self):
        pass

    def detect_communities(self, global_graph):
        """
        Analyzes the global graph to find dense neighborhoods (communities),
        which represent distinct AI sub-fields.
        Returns a mapping of node -> topic_id, and the list of communities.
        """
        # Convert directed graph to undirected for standard community detection
        undirected_g = global_graph.to_undirected()
        
        # We can only perform community detection if the graph has nodes
        if undirected_g.number_of_nodes() == 0:
            return {}, []
            
        communities = list(greedy_modularity_communities(undirected_g))
        
        node_to_community = {}
        for i, community_nodes in enumerate(communities):
            topic_id = f"Topic_{i+1}"
            for node in community_nodes:
                node_to_community[node] = topic_id
                
        return node_to_community, communities
