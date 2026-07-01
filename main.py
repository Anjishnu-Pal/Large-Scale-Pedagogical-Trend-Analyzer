import os
import networkx as nx

# Import each of the 6 steps
from step1_ingestion import DataIngestor, BaseDataSource, ManualPaper_Source, ArXivAPI_Source, KaggleLocal_Source
from step2_extraction import SemanticExtractor
from step3_knowledge_graph import KnowledgeGraphBuilder
from step4_community_detection import CommunityDetector
from step5_forecasting import TrendAnalyzer, TrendForecaster

def main():
    print("=== Large-Scale Pedagogical Trend Analyzer Pipeline ===")
    
    # ---------------------------------------------------------
    # Step 1: Ingestion
    # ---------------------------------------------------------
    print("\n[Step 1] Polymorphic Data Ingestion...")
    
    # 1. Fetch Manual Papers
    manual_papers = [
        {"title": "Efficient Fine-Tuning", "abstract": "LoRA optimizes LLMs by reducing parameters. It improves training speed significantly.", "year": 2021},
        {"title": "Advances in NLP", "abstract": "Transformers revolutionized natural language processing.", "year": 2017}
    ]
    source1 = ManualPaper_Source(papers=manual_papers)
    data1 = DataIngestor(source=source1).ingest_and_normalize()
    
    # 2. Fetch Live ArXiv Papers (Latest AI research)
    source2 = ArXivAPI_Source(query="cat:q-bio.GN", max_results=200)
    data2 = DataIngestor(source=source2).ingest_and_normalize()
    
    # 3. Fetch Offline Kaggle Data (Historical snapshot)
    source3 = KaggleLocal_Source(file_path="raw_data/arxiv-metadata-oai-snapshot.json", max_records=2)
    data3 = DataIngestor(source=source3).ingest_and_normalize()
    
    # Combine all data into one master list
    data = data1 + data2 + data3
    print(f"         Successfully combined and ingested {len(data)} total records from all 3 sources.")
    
    
    # ---------------------------------------------------------
    # Step 2: Extraction
    # ---------------------------------------------------------
    print("\n[Step 2] Semantic Information Extraction...")
    extractor = SemanticExtractor()
    processed_data = extractor.process_data(data)
    
    # ---------------------------------------------------------
    # Step 3: Knowledge Graph Construction
    # ---------------------------------------------------------
    print("\n[Step 3] Temporal Knowledge Graph Construction...")
    kg_builder = KnowledgeGraphBuilder()
    temporal_graphs = kg_builder.build_temporal_graphs(processed_data)
    
    # ---------------------------------------------------------
    # Step 4: Community Detection
    # ---------------------------------------------------------
    print("\n[Step 4] Community Detection (Topic Discovery)...")
    global_graph = nx.DiGraph()
    for g in temporal_graphs.values():
        global_graph.add_edges_from(g.edges(data=True))
        
    detector = CommunityDetector()
    node_to_community, communities, community_labels = detector.detect_communities(global_graph)
    print(f"         Detected {len(communities)} distinct AI sub-fields.")
    
    # ---------------------------------------------------------
    # Step 5: Forecasting
    # ---------------------------------------------------------
    print("\n[Step 5] Historical Aggregation and Predictive Forecasting...")
    analyzer = TrendAnalyzer()
    forecaster = TrendForecaster()
    historical_df = analyzer.calculate_historical_share(
        temporal_graphs,
        node_to_community,
        community_labels,
        processed_data=processed_data,
    )
    final_trends_df = forecaster.forecast_trends(historical_df, future_years=3, method="arima")
    
    final_trends_df.to_csv('trends_output.csv', index=False)
    print("         Data exported to 'trends_output.csv'.")
    
    # ---------------------------------------------------------
    # Step 6: Reactive UI Rendering
    # ---------------------------------------------------------
    print("\n[Step 6] Reactive UI Rendering...")
    print("         Launching Streamlit Dashboard...")
    # Execute the streamlit app directly from main
    os.system("streamlit run step6_dashboard.py")

if __name__ == "__main__":
    main()
