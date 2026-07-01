import networkx as nx
import json
import re
import os
import tempfile
import torch
from abc import ABC, abstractmethod
from networkx.algorithms.community import greedy_modularity_communities
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class CommunityDetectionStrategy(ABC):
    @abstractmethod
    def detect(self, graph):
        raise NotImplementedError


class LouvainCommunityDetectionStrategy(CommunityDetectionStrategy):
    def detect(self, graph):
        communities = list(greedy_modularity_communities(graph.to_undirected()))
        return communities


class PretrainedEmbeddingCommunityDetectionStrategy(CommunityDetectionStrategy):
    """
    Uses a pretrained GraphSAGE checkpoint to encode graph structure before
    community detection. No model training happens here.
    """
    def __init__(self, checkpoint_name="Deep696/GraphSAGE_best_model.pt"):
        self.checkpoint_name = checkpoint_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = self._build_pretrained_model()
            self.model.eval()
        except Exception:
            self.model = None

    def _build_pretrained_model(self):
        class PretrainedGraphSAGE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(5, 64)
                self.bn1 = torch.nn.BatchNorm1d(64)
                self.conv2 = SAGEConv(64, 1)

            def encode(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = self.bn1(x)
                x = torch.relu(x)
                return x

            def forward(self, x, edge_index):
                x = self.encode(x, edge_index)
                x = self.conv2(x, edge_index)
                return x

        model = PretrainedGraphSAGE()
        checkpoint_path = self._download_checkpoint()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model

    def _download_checkpoint(self):
        checkpoint_dir = os.path.join(tempfile.gettempdir(), "large_scale_trend_analyzer")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "GraphSAGE_best_model.pt")
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
            return checkpoint_path

        checkpoint_url = f"https://huggingface.co/{self.checkpoint_name}/resolve/main/GraphSAGE_best_model.pt"
        from urllib.request import urlretrieve

        urlretrieve(checkpoint_url, checkpoint_path)
        return checkpoint_path

    def _node_feature_matrix(self, graph):
        undirected_graph = graph.to_undirected()
        nodes = list(graph.nodes())
        if not nodes:
            return nodes, torch.empty((0, 5), dtype=torch.float32)

        degree_centrality = nx.degree_centrality(undirected_graph)
        pagerank = nx.pagerank(undirected_graph) if undirected_graph.number_of_edges() > 0 else {node: 0.0 for node in nodes}
        clustering = nx.clustering(undirected_graph) if undirected_graph.number_of_edges() > 0 else {node: 0.0 for node in nodes}
        max_in_degree = max((graph.in_degree(node) for node in nodes), default=1) or 1
        max_out_degree = max((graph.out_degree(node) for node in nodes), default=1) or 1

        feature_rows = []
        for node in nodes:
            in_degree = graph.in_degree(node) / max_in_degree
            out_degree = graph.out_degree(node) / max_out_degree
            total_degree = undirected_graph.degree(node) / max(1, undirected_graph.number_of_nodes() - 1)
            feature_rows.append([
                float(degree_centrality.get(node, 0.0)),
                float(in_degree),
                float(out_degree),
                float(pagerank.get(node, 0.0)),
                float(clustering.get(node, 0.0)),
            ])

        return nodes, torch.tensor(feature_rows, dtype=torch.float32)

    def _build_data_object(self, graph):
        nodes, features = self._node_feature_matrix(graph)
        if not nodes:
            return None, {}

        node_index = {node: idx for idx, node in enumerate(nodes)}
        edge_pairs = []
        for source, target in graph.edges():
            if source in node_index and target in node_index:
                edge_pairs.append([node_index[source], node_index[target]])
                edge_pairs.append([node_index[target], node_index[source]])

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=features, edge_index=edge_index), node_index

    def _node_embeddings(self, graph):
        if self.model is None or graph.number_of_nodes() == 0:
            return {}

        data, node_index = self._build_data_object(graph)
        if data is None:
            return {}

        with torch.no_grad():
            embeddings = self.model.encode(data.x, data.edge_index)

        return {
            node: embeddings[index].detach().cpu()
            for node, index in node_index.items()
        }

    def _weighted_similarity_graph(self, graph, blend_with_structure=False):
        weighted_graph = graph.to_undirected().copy()
        node_embeddings = self._node_embeddings(graph)

        if not node_embeddings:
            for source, target in weighted_graph.edges():
                weighted_graph[source][target]["weight"] = 1.0
            return weighted_graph

        with torch.no_grad():
            for source, target in weighted_graph.edges():
                source_embedding = node_embeddings.get(source)
                target_embedding = node_embeddings.get(target)
                if source_embedding is None or target_embedding is None:
                    continue
                similarity = torch.nn.functional.cosine_similarity(
                    source_embedding.unsqueeze(0),
                    target_embedding.unsqueeze(0),
                ).item()
                structural_bonus = 0.0
                if blend_with_structure:
                    common_neighbors = len(set(weighted_graph.neighbors(source)).intersection(weighted_graph.neighbors(target)))
                    structural_bonus = common_neighbors / max(1, min(weighted_graph.degree(source), weighted_graph.degree(target)))
                weighted_graph[source][target]["weight"] = 1.0 + max(0.0, 0.7 * similarity + 0.3 * structural_bonus)

        return weighted_graph

    def detect(self, graph):
        weighted_graph = self._weighted_similarity_graph(graph, blend_with_structure=False)
        return list(greedy_modularity_communities(weighted_graph, weight="weight"))


class CombinedGnnLouvainStrategy(PretrainedEmbeddingCommunityDetectionStrategy):
    """
    Combined strategy that blends pretrained GNN embeddings with graph
    structure before Louvain-style community detection.
    """
    def detect(self, graph):
        weighted_graph = self._weighted_similarity_graph(graph, blend_with_structure=True)
        return list(greedy_modularity_communities(weighted_graph, weight="weight"))

class CommunityDetector:
    TECHNICAL_HINTS = {
        "model", "network", "neural", "transformer", "attention",
        "embedding", "retrieval", "classifier", "diffusion", "graph",
        "language", "vision", "adaptation", "optimization", "architecture",
        "inference", "representation", "reinforcement", "generation", "forecast",
        "knowledge", "foundation", "ranking", "semantic", "policy", "agent",
        "adapter", "llm", "gnn", "cnn", "vit", "rlhf", "rag",
    }

    GENERIC_WORDS = {
        "we", "that", "them", "they", "this", "these", "those", "which",
        "what", "where", "when", "why", "how", "benefits", "properties",
        "results", "methods", "agreement", "advances", "energy", "place",
        "calculation", "approach", "paper", "signals", "sources", "rules",
        "tools", "us", "n", "li", "it", "people", "things", "problem",
        "model", "models", "training", "accuracy", "work", "works", "features",
        "feature", "actions", "action", "loop", "loops", "optimization", "adaptation",
    }

    def __init__(self, strategy=None):
        self.strategy = self._resolve_strategy(strategy)
        self.aliases = self._load_aliases()

    def _load_aliases(self):
        try:
            with open("aliases.json", "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _normalize_label(self, text):
        normalized = str(text).replace("_", " ").strip().lower()
        return re.sub(r"\s+", " ", normalized)

    def _is_research_term(self, text):
        normalized = self._normalize_label(text)
        if not normalized:
            return False

        if normalized in self.aliases.values() or normalized in self.aliases.keys():
            return True

        tokens = normalized.split()
        if len(tokens) == 1 and text.isupper() and 2 <= len(text) <= 6:
            return True

        if len(tokens) == 1:
            return False

        if any(token in self.TECHNICAL_HINTS for token in tokens):
            return True

        if "language model" in normalized or "neural network" in normalized or "graph neural" in normalized:
            return True

        return False

    def _label_score(self, text, centrality_score):
        normalized = self._normalize_label(text)
        if not normalized:
            return -1.0

        if any(word in normalized.split() for word in self.GENERIC_WORDS):
            return -1.0

        score = float(centrality_score)
        if normalized in self.aliases.values() or normalized in self.aliases.keys():
            score += 10.0
        if text.isupper() and 2 <= len(text) <= 6:
            score += 8.0
        if any(token in self.TECHNICAL_HINTS for token in normalized.split()):
            score += 6.0
        if "language model" in normalized or "neural network" in normalized or "graph neural" in normalized:
            score += 6.0
        if len(normalized.split()) >= 2:
            score += 1.0
        else:
            score -= 4.0
        return score

    def _compose_research_label(self, ranked_nodes):
        research_candidates = []
        fallback_candidates = []

        for node, degree in ranked_nodes:
            pretty_node = str(node).replace("_", " ").strip()
            if not pretty_node:
                continue
            score = self._label_score(pretty_node, degree)
            if score < 0:
                continue

            normalized = self._normalize_label(pretty_node)
            if self._is_research_term(pretty_node):
                research_candidates.append((score, normalized))
            else:
                fallback_candidates.append((score, normalized))

        if research_candidates:
            best_label = max(research_candidates, key=lambda item: item[0])[1]
            cleaned = self.aliases.get(best_label, best_label)
            return cleaned.replace("_", " ").title()

        if fallback_candidates:
            top_terms = []
            seen = set()
            for _, normalized in sorted(fallback_candidates, key=lambda item: item[0], reverse=True):
                if normalized in seen:
                    continue
                seen.add(normalized)
                if len(normalized.split()) == 1 and normalized not in self.GENERIC_WORDS and len(normalized) > 3:
                    top_terms.append(normalized)
                if len(top_terms) == 2:
                    break

            if top_terms:
                joined = " / ".join(term.replace("_", " ").title() for term in top_terms)
                return joined

        return "Unknown Topic"

    def _community_label_candidates(self, graph, community_nodes):
        subgraph = graph.subgraph(community_nodes).to_undirected()
        candidates = []
        for node, degree in sorted(subgraph.degree(), key=lambda item: (-item[1], item[0])):
            label = str(node).replace("_", " ").strip()
            normalized = self._normalize_label(label)
            if not normalized or normalized in self.GENERIC_WORDS:
                continue
            if len(normalized) <= 2:
                continue
            if any(char.isdigit() for char in normalized) and len(normalized) < 4:
                continue
            if len(normalized.split()) == 1 and normalized not in self.aliases and normalized not in self.aliases.values():
                continue
            score = self._label_score(label, degree)
            if score < 0:
                continue
            candidates.append((score, normalized))
        return candidates

    def _resolve_strategy(self, strategy):
        if strategy is None:
            return LouvainCommunityDetectionStrategy()

        if isinstance(strategy, CommunityDetectionStrategy):
            return strategy

        if isinstance(strategy, str):
            normalized = strategy.strip().lower()
            if normalized == "louvain":
                return LouvainCommunityDetectionStrategy()
            if normalized in {"gnn", "combined", "combined_gnn_louvain", "embedding"}:
                return CombinedGnnLouvainStrategy()

        return LouvainCommunityDetectionStrategy()

    def _build_topic_label(self, graph, community_nodes):
        """
        Build a human-readable research-topic label from the most central nodes in a community.
        """
        subgraph = graph.subgraph(community_nodes).to_undirected()
        if subgraph.number_of_nodes() == 0:
            return "Unknown Topic"

        ranked_nodes = sorted(
            subgraph.degree(),
            key=lambda item: (-item[1], item[0])
        )

        candidates = self._community_label_candidates(graph, community_nodes)
        if candidates:
            best_label = max(candidates, key=lambda item: item[0])[1]
            cleaned = self.aliases.get(best_label, best_label)
            return cleaned.replace("_", " ").title()

        return self._compose_research_label(ranked_nodes)

    def detect_communities(self, global_graph):
        """
        Analyzes the global graph to find dense neighborhoods (communities),
        which represent distinct AI sub-fields.
        Returns a mapping of node -> topic_id, the list of communities,
        and a mapping of topic_id -> human-readable topic label.
        """
        # We can only perform community detection if the graph has nodes
        if global_graph.number_of_nodes() == 0:
            return {}, [], {}

        communities = self.strategy.detect(global_graph)
        
        node_to_community = {}
        community_labels = {}
        for i, community_nodes in enumerate(communities):
            topic_id = f"Topic_{i+1}"
            community_labels[topic_id] = self._build_topic_label(global_graph.to_undirected(), community_nodes)
            for node in community_nodes:
                node_to_community[node] = topic_id
                
        return node_to_community, communities, community_labels
