# Large-Scale Pedagogical Trend Analyzer

The Large-Scale Pedagogical Trend Analyzer is an advanced, interdisciplinary project designed to track and predict the evolution of different fields in research. Every year, thousands of academic papers are published, making it impossible for educators and researchers to manually track which technologies are emerging and which are becoming obsolete.

This project solves that problem by building an automated, event-driven data pipeline. It ingests thousands of unstructured research abstracts, extracts the core concepts using Natural Language Processing (NLP), and maps them into a Temporal Knowledge Graph. By applying network science and Graph Representation Learning the system identifies distinct academic sub-fields, tracks their historical growth and time-series forecasting mathematically predicts where the research frontier will move next.

## Project Architecture & Steps

### Step 1: Polymorphic Data Ingestion
The system connects to web sources or local directories to pull raw academic metadata (titles, abstracts, publication years) and normalizes the text. We utilize a strict Abstract Base Class (`BaseDataSource`) for polymorphism, supporting:
1. `ArXivAPI_Source`: Live streaming data.
2. `KaggleLocal_Source`: Batch processing massive JSON snapshots.
3. `ManualPaper_Source`: Targeted manual injection.

**File**: `step1_ingestion.py`

### Step 2: Semantic Information Extraction
Instead of simple keyword matching, an NLP relation extraction model scans the text to pull logical triplets (Subject-Predicate-Object).
**File**: `step2_extraction.py`

### Step 3: Temporal Knowledge Graph Construction
The extracted concepts are assembled into a mathematical network of nodes and directed edges, partitioned by year to analyze trends over time. 
**File**: `step3_knowledge_graph.py`

### Step 4: Community Detection (Topic Discovery)
The system analyzes the graph to find dense neighborhoods of connected concepts, automatically identifying distinct AI sub-fields without human labeling. 
**File**: `step4_community_detection.py`

### Step 5: Historical Aggregation and Predictive Forecasting
The system calculates the historical market share of each detected community over time, then applies mathematical models (like Linear Regression) to extend those trajectories into the future. 
**File**: `step5_forecasting.py`

### Step 6: Reactive UI Rendering
The final historical and predicted data points are visualized on an interactive time-series dashboard using Streamlit and Plotly. 
**File**: `step6_dashboard.py`

## Getting Started

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the complete pipeline:**
   ```bash
   python main.py
   ```
   *This single command will execute Steps 1-5, generate the forecasted data (`trends_output.csv`), and automatically launch the Streamlit dashboard in your browser for Step 6.*
