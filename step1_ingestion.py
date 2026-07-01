import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod


class BaseDataSource(ABC):
    """Abstract base interface for all data sources."""

    @abstractmethod
    def fetch_data(self) -> list:
        """Return a list of dictionaries with title, abstract, and year."""


class ArXivAPI_Source(BaseDataSource):
    """Live streaming data source backed by the arXiv API."""

    def __init__(self, query="cat:cs.LG OR cat:cs.AI", max_results=10):
        self.query = query.replace(" ", "+")
        self.max_results = max_results
        self.base_url = "http://export.arxiv.org/api/query?"

    def fetch_data(self):
        print(f"--> [ArXivAPI_Source] Fetching live data for: {self.query}")
        url = (
            f"{self.base_url}search_query={self.query}"
            f"&max_results={self.max_results}&sortBy=submittedDate&sortOrder=descending"
        )

        try:
            response = urllib.request.urlopen(url)
            xml_data = response.read()
            root = ET.fromstring(xml_data)

            data = []
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.replace("\n", " ")
                abstract = entry.find("atom:summary", ns).text.replace("\n", " ")
                published = entry.find("atom:published", ns).text
                year = int(published[:4])
                data.append({"title": title.strip(), "abstract": abstract.strip(), "year": year})
            return data
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")
            return []


class KaggleLocal_Source(BaseDataSource):
    """Batch/offline source that reads the local Kaggle arXiv JSON snapshot."""

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
            with open(self.file_path, "r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index >= self.max_records:
                        break

                    record = json.loads(line)
                    title = record.get("title", "")
                    abstract = record.get("abstract", "")
                    date_str = record.get("update_date", "2000-01-01")
                    year = int(date_str.split("-")[0])

                    data.append({"title": title.strip(), "abstract": abstract.strip(), "year": year})
            return data
        except Exception as e:
            print(f"Error reading local Kaggle data: {e}")
            return []


class ManualPaper_Source(BaseDataSource):
    """Manual source for injecting a small set of provided papers."""

    def __init__(self, papers: list):
        self.papers = papers

    def fetch_data(self):
        print(f"--> [ManualPaper_Source] Using {len(self.papers)} manually submitted papers")
        return self.papers


class DataIngestor:
    """Normalizes data fetched from any BaseDataSource implementation."""

    def __init__(self, source: BaseDataSource):
        self.source = source

    def normalize_text(self, text):
        if not isinstance(text, str):
            return ""
        return " ".join(text.split()).lower()

    def ingest_and_normalize(self):
        raw_data = self.source.fetch_data()

        for item in raw_data:
            if "title" in item:
                item["title_normalized"] = self.normalize_text(item["title"])
            if "abstract" in item:
                item["abstract_normalized"] = self.normalize_text(item["abstract"])

        return raw_data
