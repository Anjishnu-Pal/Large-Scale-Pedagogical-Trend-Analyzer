import spacy
import json
import requests
import urllib.parse
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
            
            # 2. Extract from Semantic Scholar Citation Contexts!
            # What do OTHER papers say about this one?
            citation_contexts = self.citation_fetcher.fetch_contexts(title)
            for ctx in citation_contexts:
                ctx_triplets = self._extract_from_text(ctx)
                triplets.extend(ctx_triplets)
            
            # Save our list of Relationship objects
            item['triplet_objects'] = triplets
            results.append(item)
            
        return results

if __name__ == "__main__":
    extractor = SemanticExtractor()
    sample = [{"title": "Sparsity-certifying Graph Decompositions", "abstract_normalized": "we describe a new algorithm, the $(k,\\ell)$-pebble game with colors, and use it obtain a characterization of the family of $(k,\\ell)$-sparse graphs and algorithmic solutions to a family of problems concerning tree decompositions of graphs. special instances of sparse graphs appear in rigidity theory and have received increased attention in recent years. in particular, our colored pebbles generalize and strengthen the previous results of lee and streinu and give a new proof of the tutte-nash-williams characterization of arboricity. we also present a new decomposition that certifies sparsity based on the $(k,\\ell)$-pebble game with colors. our work also exposes connections between pebble game algorithms and previous sparse graph algorithms by gabow, gabow and westermann and hendrickson."}]
    res = extractor.process_data(sample)
    print(f"Extracted Triplets: {res[0]['triplet_objects']}")
