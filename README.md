# Mapping Knowledge: Collaboration, Topics, and Influence in the DBLP Citation Network

**Group 7**

## 📌 Project Overview
This project aims to understand the evolution of computer science research using the DBLP dataset. We will analyze collaboration trends, rising research topics, and patterns of influence by building citation and co-authorship networks. The project involves a scalable ETL pipeline, network analysis, topic modeling, and predictive modeling.

## 👥 Team & Roles

| Member | Role | Focus Areas |
|--------|------|-------------|
| **Truc Le** | Data Engineering & Infrastructure | Ingestion, Schema Normalization, Parquet Storage, Edge Building (Citations/Co-authors) |
| **Ai Nhien To** | Networks & Metrics | Graph Construction, Centrality/Community Computation, Temporal Slicing, Influence Trajectories |
| **Julio Amaya** | NLP & Modeling | Text Cleaning, Topic Modeling (TF-IDF/LDA), Trend Analysis, Predictive Modeling (Citation Impact) |

## 📂 Project Structure

```
COSC-3337-Project/
├── data/
│   ├── raw/                  # Raw DBLP JSON shards (not committed)
│   └── parquet/              # Cleaned Parquet datasets
│       ├── papers/           # Core publication metadata
│       ├── authorships/      # Author-paper relationships
│       ├── citations/        # Citation network edges
│       └── coauthorships/    # Coauthor collaboration edges
├── notebooks/                # Jupyter Notebooks for Analysis
│   ├── 01_data_engineering_etl.ipynb      # ETL Pipeline (Truc)
│   ├── 02_data_profiling_analysis.ipynb   # Data Profiling (Truc)
│   ├── 03_network_analysis.ipynb          # Network metrics & graphs (Ai Nhien)
│   ├── 04_predictive_modeling.ipynb       # Predictive modeling (Julio)
│   ├── 05_anomaly_detection.ipynb         # Anomaly detection
│   └── 06_topic_modeling.ipynb            # Topic modeling (Julio)
├── src/                      # Source Code Modules
│   ├── etl/                  # Data Engineering (Truc)
│   │   ├── ingestion.py      # JSON streaming & parsing
│   │   └── processing.py     # Cleaning & Parquet conversion
│   ├── networks/             # Network Analysis (Ai Nhien)
│   │   ├── graph_builder.py  # NetworkX/igraph construction
│   │   └── metrics.py        # Centrality & Community algorithms
│   ├── nlp_modeling/         # NLP & ML (Julio)
│   │   ├── text_processing.py # TF-IDF, Cleaning
│   │   └── models.py         # LDA, Classifiers (LogReg, XGBoost)
│   └── utils/                # Shared utilities
│       └── config.py         # Paths and constants
├── docs/
│   └── data_dictionary.md    # Comprehensive data documentation
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

## 🚀 Getting Started

### 1. Environment Setup

**Prerequisites:**
- Python 3.13.1 or higher (required for all dependencies to work correctly)

Create a virtual environment and install dependencies:
```bash
python --version  # Verify you have Python 3.13.1+
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you encounter a `ModuleNotFoundError` for `networkx`, install it manually:
```bash
pip install networkx
```

### 2. Data Processing 
The ETL pipeline has been successfully executed. All cleaned datasets are available in `data/parquet/`:
- Papers metadata with 9 columns
- Authorships with normalized author names
- Citations network edges
- Coauthorships network edges

See `docs/data_dictionary.md` for detailed schema documentation.

### 3. Analysis Workflow
- **Data Profiling**: Review `notebooks/02_data_profiling_analysis.ipynb` for dataset statistics
- **Networks**: Use `notebooks/03_network_analysis.ipynb` for graph analysis (Ai Nhien)
- **NLP/ML**: Use `notebooks/04_predictive_modeling.ipynb` and `06_topic_modeling.ipynb` (Julio)

## Project Progress & Roadmap

### Phase 1: Data Engineering (Truc) 
- [x] Implement chunked stream-parsing for JSON shards
- [x] Clean data: Drop missing IDs, normalize venues, handle missing abstracts
- [x] Build citation network edges (directed graph)
- [x] Build coauthorship network edges (undirected graph)
- [x] Export to Parquet: `papers`, `authorships`, `citations`, `coauthorships`
- [x] Create comprehensive data dictionary (377 lines)
- [x] Data profiling and quality analysis notebook
- [x] Schema normalization with author name standardization

### Phase 2: Network Analysis (Nhien) 
- [x] Build Citation Graph (Directed) & Co-authorship Graph (Undirected)
- [x] Compute Centralities: Degree, PageRank, Betweenness
- [x] Detect Communities (Louvain) and track temporal evolution
- [x] Generate network visualizations and metrics tables

### Phase 3: NLP & Modeling (Julio) 
- [ ] Text Features: TF-IDF (10K features) -> PCA
- [ ] Topic Modeling: LDA/NMF to identify subfields
- [ ] Predictive Task: Forecast citation impact (Pre-2010 train / Post-2010 test)
- [ ] Anomaly detection in publications and collaborations

### Phase 4: Integration & Reporting (All) 
- [ ] Combine insights from all analyses
- [ ] Create publication-ready figures and tables
- [ ] Write final report with findings and recommendations
- [ ] Prepare presentation materials
