import networkx as nx
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeConcept:
    """Represents a conceptual node in the knowledge graph."""
    name: str


@dataclass(frozen=True)
class EdgeRelation:
    """Represents a directed edge between two conceptual nodes."""
    source: NodeConcept
    target: NodeConcept
    relation: str


class YearlyGraph:
    """
    A graph for a single year composed of NodeConcept and EdgeRelation objects.
    The internal NetworkX graph is kept in sync for downstream compatibility.
    """
    def __init__(self, year):
        self.year = int(year)
        self.node_objects = {}
        self.edge_objects = []
        self._graph = nx.DiGraph()

    def add_node(self, node_name):
        node_key = str(node_name)
        if node_key not in self.node_objects:
            self.node_objects[node_key] = NodeConcept(name=node_key)
            self._graph.add_node(node_key)
        return self.node_objects[node_key]

    def add_edge(self, source_name, target_name, relation):
        source_node = self.add_node(source_name)
        target_node = self.add_node(target_name)
        edge = EdgeRelation(source=source_node, target=target_node, relation=str(relation))
        self.edge_objects.append(edge)
        self._graph.add_edge(source_node.name, target_node.name, relation=edge.relation)
        return edge

    @property
    def graph(self):
        return self._graph

    @property
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        return self._graph.edges

    def number_of_nodes(self):
        return self._graph.number_of_nodes()

    def to_undirected(self):
        return self._graph.to_undirected()

    def __len__(self):
        return self._graph.number_of_nodes()


class TemporalKnowledgeGraph:
    """
    Master object composed of yearly graph objects.
    This preserves the year-partitioned structure used throughout the pipeline.
    """
    def __init__(self):
        self.yearly_graphs = {}

    def get_or_create_year_graph(self, year):
        year_key = int(year)
        if year_key not in self.yearly_graphs:
            self.yearly_graphs[year_key] = YearlyGraph(year_key)
        return self.yearly_graphs[year_key]

    def add_relation(self, year, source_name, target_name, relation):
        year_graph = self.get_or_create_year_graph(year)
        year_graph.add_edge(source_name, target_name, relation)

    def items(self):
        return self.yearly_graphs.items()

    def values(self):
        return self.yearly_graphs.values()

    def keys(self):
        return self.yearly_graphs.keys()

    def __getitem__(self, year):
        return self.yearly_graphs[year]

    def __contains__(self, year):
        return year in self.yearly_graphs

    def __iter__(self):
        return iter(self.yearly_graphs)

class KnowledgeGraphBuilder:
    def __init__(self):
        pass

    def build_temporal_graphs(self, processed_data):
        """
        Builds a master temporal graph composed of yearly graphs.
        Input: List of documents containing 'year' and extracted 'triplet_objects'
        Output: TemporalKnowledgeGraph composed of YearlyGraph objects
        """
        temporal_graph = TemporalKnowledgeGraph()
        
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
                    # We add edges per year inside the composed yearly graph
                    temporal_graph.add_relation(year, subj, obj, pred)
                    
        return temporal_graph
