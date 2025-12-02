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

## 📂 Project Structure (Guideline Map)

```
COSC-3337-Project/
├── data/
│   ├── raw/                  # Place dblp-ref-*.json shards here
│   └── processed/            # Output Parquet files (papers, authorships, citations)
├── notebooks/                # Jupyter Notebooks for Analysis & EDA
│   ├── 01_eda.ipynb          # Exploratory Data Analysis (Shared)
│   ├── 02_network_analysis.ipynb # Network metrics & graphs (Ai Nhien)
│   ├── 03_topic_modeling.ipynb   # Topic modeling & trends (Julio)
│   ├── 04_predictive_modeling.ipynb # Classification & Prediction (Julio)
│   └── 05_anomaly_detection.ipynb # Outlier detection
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
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

## 🚀 Getting Started

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Ingestion (Truc)
Place the DBLP JSON shards in `data/raw/`. Run the ETL pipeline to generate Parquet files:
```bash
python src/etl/ingestion.py
```

### 3. Analysis Workflow
- **EDA**: Run `notebooks/01_eda.ipynb` to visualize distributions.
- **Networks**: Use `src/networks/` modules in `notebooks/02_network_analysis.ipynb`.
- **NLP/ML**: Use `src/nlp_modeling/` modules in `notebooks/03_topic_modeling.ipynb` and `04_predictive_modeling.ipynb`.

## ✅ Key Tasks & Roadmap

### Phase 1: Data Engineering (Truc)
- [ ] Implement chunked stream-parsing for JSON shards.
- [ ] Clean data: Drop missing IDs, normalize venues, handle missing abstracts.
- [ ] Export to Parquet: `papers.parquet`, `authorships.parquet`, `citations.parquet`.

### Phase 2: Network Analysis (Nhien)
- [ ] Build Citation Graph (Directed) & Co-authorship Graph (Undirected).
- [ ] Compute Centralities: Degree, PageRank, Betweenness.
- [ ] Detect Communities (Louvain) and track temporal evolution.

### Phase 3: NLP & Modeling (Julio)
- [ ] Text Features: TF-IDF (10K features) -> PCA.
- [ ] Topic Modeling: LDA/NMF to identify subfields.
- [ ] Predictive Task: Forecast citation impact (Pre-2010 train / Post-2010 test).

### Phase 4: Integration & Reporting (All)
- [ ] Combine features into a master dataset.
- [ ] Finalize visualizations and report.
