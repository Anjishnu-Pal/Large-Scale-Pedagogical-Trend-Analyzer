import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

# ---------------------------------------------------------
# 1. Strict Abstract Base Interface
# ---------------------------------------------------------
class BaseDataSource(ABC):
    """
    Abstract Base Class for all data sources.
    Forces every source to implement the `fetch_data` method.
    """
    @abstractmethod
    def fetch_data(self) -> list:
        """
        Must return a list of dictionaries, each containing:
        - title (str)
        - abstract (str)
        - year (int)
        """
        pass

# ---------------------------------------------------------
# 2. Concrete Subclasses
# ---------------------------------------------------------
class ArXivAPI_Source(BaseDataSource):
    """
    Live Streaming Data: Connects to the open arXiv API dynamically.
    Streams latest publications in specific CS sub-categories (e.g. CS.LG).
    """
    def __init__(self, query="cat:cs.LG OR cat:cs.AI", max_results=10):
        self.query = query.replace(" ", "+")
        self.max_results = max_results
        self.base_url = "http://export.arxiv.org/api/query?"

    def fetch_data(self):
        print(f"--> [ArXivAPI_Source] Fetching live data for: {self.query}")
        url = f"{self.base_url}search_query={self.query}&max_results={self.max_results}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            response = urllib.request.urlopen(url)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            data = []
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.replace('\n', ' ')
                abstract = entry.find('atom:summary', ns).text.replace('\n', ' ')
                published = entry.find('atom:published', ns).text
                year = int(published[:4])
                data.append({"title": title.strip(), "abstract": abstract.strip(), "year": year})
            return data
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")
            return []

class KaggleLocal_Source(BaseDataSource):
    """
    Batch/Offline Data: Reads from the massive official Kaggle arXiv Dataset (JSON).
    Tests memory management and historical processing without web limits.
    """
    def __init__(self, file_path, max_records=1000):
        self.file_path = file_path
        self.max_records = max_records

    def fetch_data(self):
        print(f"--> [KaggleLocal_Source] Reading batch data from: {self.file_path}")
        data = []
        if not os.path.exists(self.file_path):
            print(f"Warning: Kaggle JSON snapshot '{self.file_path}' not found. Returning empty.")
            return data
            
        try:
            # Process large JSON file line-by-line for memory management
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= self.max_records:
                        break
                    record = json.loads(line)
                    title = record.get('title', '')
                    abstract = record.get('abstract', '')
                    
                    # Kaggle metadata uses 'update_date' like "2008-11-26"
                    date_str = record.get('update_date', '2000-01-01')
                    year = int(date_str.split('-')[0])
                    
                    data.append({"title": title.strip(), "abstract": abstract.strip(), "year": year})
            return data
        except Exception as e:
            print(f"Error reading local Kaggle data: {e}")
            return []

class ManualPaper_Source(BaseDataSource):
    """
    Manual Data: Injects specific manually provided papers.
    """
    def __init__(self, papers: list):
        self.papers = papers

    def fetch_data(self):
        print(f"--> [ManualPaper_Source] Using {len(self.papers)} manually submitted papers")
        return self.papers

# ---------------------------------------------------------
# 3. Main Ingestor (Uses Polymorphism)
# ---------------------------------------------------------
class DataIngestor:
    """
    The main system pulls data from any source strictly via the `BaseDataSource` interface, 
    without needing to know its underlying network/file mechanics.
    """
    def __init__(self, source: BaseDataSource):
        self.source = source

    def normalize_text(self, text):
        """Standardizes text for NLP processing."""
        if not isinstance(text, str):
            return ""
        return " ".join(text.split()).lower()

    def ingest_and_normalize(self):
        # Polymorphic call - we don't care if it's API, File, or Manual!
        raw_data = self.source.fetch_data()
        
        # Normalization step
        for item in raw_data:
            if 'title' in item:
                item['title_normalized'] = self.normalize_text(item['title'])
            if 'abstract' in item:
                item['abstract_normalized'] = self.normalize_text(item['abstract'])
                
        return raw_data

if __name__ == "__main__":
    import json
    print("=== Testing Manual Source ===")
    manual_papers = [
        {"title": "AI Flow: perspectives, scenarios, and approaches", "abstract": "Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems. These advancements are crucial to numerous application scenarios, including but not limited to embodied AI, wearable devices, and smart cities.", "year": 2026}
    ]
    source = ManualPaper_Source(papers=manual_papers)
    ingestor = DataIngestor(source)
    data = ingestor.ingest_and_normalize()
    print(json.dumps(data, indent=2))

    print("\n=== Testing ArXiv API Source ===")
    # Change 'max_results' to get more papers, and 'query' to change the topic!
    api_source = ArXivAPI_Source(query="cat:cs.CV", max_results=10)
    api_ingestor = DataIngestor(api_source)
    api_data = api_ingestor.ingest_and_normalize()
    print(json.dumps(api_data, indent=2))

    print("\n=== Testing Kaggle Source ===")
    # Reads only the first 2 records to test quickly without crashing memory
    kaggle_source = KaggleLocal_Source(file_path="raw_data/arxiv-metadata-oai-snapshot.json", max_records=2)
    kaggle_ingestor = DataIngestor(kaggle_source)
    kaggle_data = kaggle_ingestor.ingest_and_normalize()
    print(json.dumps(kaggle_data, indent=2))
