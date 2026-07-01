import spacy
import json
import requests
import urllib.parse
from itertools import combinations
from dataclasses import dataclass

# =========================================================
# OOP CONCEPT: ENCAPSULATION
# Data is securely wrapped in strictly typed Python objects
# rather than messy loose dictionaries or raw strings.
# =========================================================

@dataclass
class Entity:
    """Represents a clean extracted concept."""
    original_text: str
    canonical_name: str

@dataclass
class Relationship:
    """Represents a Subject-Predicate-Object triplet."""
    subject: Entity
    predicate: str
    object_: Entity  # Appended underscore as 'object' is a reserved keyword in Python


class EntityNormalizer:
    """
    Feature 1: The Canonical Alias Dictionary
    Forces variations of terms (e.g. 'CNN', 'ConvNet') to a single standard ID.
    """
    def __init__(self, alias_file="aliases.json"):
        self.aliases = {}
        try:
            with open(alias_file, 'r', encoding='utf-8') as f:
                self.aliases = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {alias_file} not found. Using empty dictionary.")

    def normalize(self, text: str) -> Entity:
        clean_text = text.lower().strip()
        # Look up in dictionary, default to the clean_text if not found
        canonical = self.aliases.get(clean_text, clean_text)
        return Entity(original_text=text, canonical_name=canonical)


class CitationContextFetcher:
    """
    Feature 2: Semantic Scholar "Citation Contexts"
    Pulls sentences from *other* papers that cited this paper to find methodology triplets.
    """
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def fetch_contexts(self, title: str, limit=1) -> list:
        if not title: return []
        # print(f"      [Semantic Scholar] Looking up citations for: {title[:40]}...")
        try:
            query = urllib.parse.quote(title)
            url = f"{self.base_url}?query={query}&fields=citations.contexts&limit={limit}"
            
            # Simple API request
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return []
                
            data = response.json()
            contexts = []
            
            if "data" in data and len(data["data"]) > 0:
                paper = data["data"][0]
                citations = paper.get("citations", [])
                for cit in citations:
                    ctx_list = cit.get("contexts", [])
                    contexts.extend(ctx_list)
            return contexts
        except Exception as e:
            return []


class SemanticExtractor:
    """
    The main coordinator. Hides all the messy text parsing, Spacy logic, 
    and API fetching from the rest of the application.
    """
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Model {model} not found. Ensure spaCy is installed.")
            self.nlp = None
            
        self.normalizer = EntityNormalizer()
        self.citation_fetcher = CitationContextFetcher()
        self.technical_hints = {
            "model", "network", "neural", "transformer", "attention", "learning",
            "embedding", "retrieval", "classifier", "diffusion", "graph", "language",
            "vision", "adaptation", "optimization", "architecture", "inference",
            "representation", "reinforcement", "generation", "knowledge", "foundation",
            "semantic", "adapter", "llm", "gnn", "cnn", "vit", "rlhf", "rag",
            "encoder", "decoder", "prompt", "tuning", "pretraining", "fine", "self",
            "supervised", "multimodal", "policy", "agent", "bayesian", "federated",
            "transfer", "contrastive", "distillation", "zero", "few", "shot",
        }
        self.generic_words = {
            "we", "that", "them", "they", "this", "these", "those", "which", "what",
            "where", "when", "why", "how", "benefits", "properties", "results", "methods",
            "agreement", "advances", "energy", "place", "calculation", "approach", "paper",
            "signals", "sources", "rules", "tools", "us", "n", "li", "it", "people", "things", "problem",
            "model", "models", "work", "study", "analysis", "system", "systems", "method", "methods",
            "results", "result", "effect", "effects", "feature", "features",
        }

    def _normalize_concept(self, text: str) -> str:
        if not text:
            return ""
        normalized = text.lower().replace("_", " ").strip()
        return " ".join(normalized.split())

    def _is_research_concept(self, text: str) -> bool:
        normalized = self._normalize_concept(text)
        if not normalized:
            return False

        if normalized in self.normalizer.aliases:
            return True

        if normalized in self.normalizer.aliases.values():
            return True

        tokens = normalized.split()
        if any(token in self.generic_words for token in tokens):
            return False

        if len(tokens) >= 2 and any(token in self.technical_hints for token in tokens):
            return True

        if len(tokens) == 1 and (text.isupper() and 2 <= len(text) <= 6):
            return True

        if "language model" in normalized or "neural network" in normalized or "graph neural" in normalized:
            return True

        if any(token in self.technical_hints for token in tokens):
            return True

        return False

    def _extract_research_concepts(self, text: str) -> list:
        if not self.nlp or not text:
            return []

        doc = self.nlp(text)
        concepts = []
        seen = set()

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip(" -,:;()[]{}\"'\n\t")
            normalized = self._normalize_concept(chunk_text)
            if len(normalized) < 3:
                continue
            if not self._is_research_concept(normalized):
                continue
            canonical = self.normalizer.normalize(chunk_text).canonical_name
            if canonical and canonical not in seen:
                seen.add(canonical)
                concepts.append(canonical)

        return concepts

    def _build_concept_cooccurrence_triplets(self, concepts: list) -> list:
        if len(concepts) < 2:
            return []

        unique_concepts = []
        seen = set()
        for concept in concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)

        limited_concepts = unique_concepts[:8]
        relationships = []
        for left, right in combinations(limited_concepts, 2):
            subj_entity = self.normalizer.normalize(left)
            obj_entity = self.normalizer.normalize(right)
            relationships.append(
                Relationship(
                    subject=subj_entity,
                    predicate="co_occurs_with",
                    object_=obj_entity,
                )
            )

        return relationships

    def _extract_from_text(self, text: str) -> list:
        """Internal private method to parse strings into Relationship objects."""
        if not self.nlp or not text:
            return []
            
        doc = self.nlp(text)
        relationships = []
        
        for token in doc:
            # Look for verbs (Predicates)
            if token.pos_ == "VERB":
                subj_text = None
                obj_text = None
                
                # Find children for Subject and Object
                for child in token.children:
                    if "subj" in child.dep_:
                        subj_text = child.text
                    if "obj" in child.dep_:
                        obj_text = child.text
                        
                if subj_text and obj_text:
                    # Encapsulate! Convert raw strings into strict Entity objects
                    subj_entity = self.normalizer.normalize(subj_text)
                    obj_entity = self.normalizer.normalize(obj_text)
                    
                    # Create the final Relationship object
                    rel = Relationship(
                        subject=subj_entity,
                        predicate=token.lemma_, # the base form of the verb (e.g. 'optimizes' -> 'optimize')
                        object_=obj_entity
                    )
                    relationships.append(rel)
                    
        return relationships

    def process_data(self, data_list: list) -> list:
        """Public method. Accepts clean dictionaries and appends strictly typed objects."""
        results = []
        for item in data_list:
            title = item.get('title', '')
            abstract = item.get('abstract_normalized', '')
            
            # 1. Extract from the abstract
            triplets = self._extract_from_text(abstract)
            
            # 1b. Extract research concepts and create co-occurrence links so the
            # graph contains topic-level nodes, not only verb fragments.
            topic_concepts = self._extract_research_concepts(title)
            topic_concepts.extend(self._extract_research_concepts(abstract))
            
            # 2. Extract from Semantic Scholar Citation Contexts!
            # What do OTHER papers say about this one?
            citation_contexts = self.citation_fetcher.fetch_contexts(title)
            for ctx in citation_contexts:
                ctx_triplets = self._extract_from_text(ctx)
                triplets.extend(ctx_triplets)
                topic_concepts.extend(self._extract_research_concepts(ctx))

            triplets.extend(self._build_concept_cooccurrence_triplets(topic_concepts))
            
            # Save our list of Relationship objects
            item['triplet_objects'] = triplets
            results.append(item)
            
        return results
