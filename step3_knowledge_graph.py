import networkx as nx
from collections import defaultdict

class KnowledgeGraphBuilder:
    def __init__(self):
        pass

    def build_temporal_graphs(self, processed_data):
        """
        Builds a dictionary of temporal graphs partitioned by year.
        Input: List of documents containing 'year' and extracted 'triplet_objects'
        Output: dict mapping year (int) to nx.DiGraph
        """
        graphs_by_year = defaultdict(nx.DiGraph)
        
        for item in processed_data:
            year = item.get('year')
            if not year:
                continue
                
            # We now safely retrieve the encapsulated Python objects!
            triplet_objects = item.get('triplet_objects', [])
            for rel in triplet_objects:
                # We pull the clean canonical names instead of messy raw text
                subj = rel.subject.canonical_name
                pred = rel.predicate
                obj = rel.object_.canonical_name
                
                # Only add if we have both subject and object to form an edge
                if subj and obj:
                    # We add edges per year
                    graphs_by_year[year].add_edge(subj, obj, relation=pred)
                    
        return dict(graphs_by_year)
