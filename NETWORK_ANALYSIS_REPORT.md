# Network Analysis Module Report

**COSC 3337 - Data Science Project**  
**Contributor:** Ai Nhien To  
**Role:** Network Analysis - Citation & Co-authorship Graphs  
**Date:** December 5, 2025

---

## 1. Executive Summary

This module conducted comprehensive **graph-theoretic analysis** of the DBLP research network, examining both **citation patterns** (knowledge flow) and **collaboration structures** (social networks). Using network science methodologies, we analyzed 3+ million papers and 1.7+ million authors to uncover the hidden structures that drive scientific influence and community formation.

**Key Achievement:** Identified 326,678 influential papers through PageRank centrality analysis and discovered 8,247 distinct research communities through Louvain clustering, revealing the organizational structure of computer science research across 82 years of publication history.

---

## 2. Project Scope & Objectives

### 2.1 Research Questions
1. **Influence:** Which papers and authors are most central to knowledge dissemination?
2. **Community:** How do researchers cluster into collaborative communities?
3. **Evolution:** How have citation and collaboration patterns changed over time?
4. **Structure:** What network properties characterize the DBLP graph (scale-free, small-world)?

### 2.2 Deliverables
1. ✅ **Citation Network:** Directed graph with 25M edges, centrality metrics
2. ✅ **Coauthor Network:** Undirected graph with 14M edges, community detection
3. ✅ **Temporal Analysis:** Network evolution across decades
4. ✅ **Centrality Rankings:** PageRank, degree, betweenness for papers/authors
5. ✅ **Community Structure:** Louvain modularity-based clustering
6. ✅ **Visualizations:** Network diagrams and statistical plots

---

## 3. Methodology

### 3.1 Graph Construction

#### 3.1.1 Citation Network (Directed Graph)

**Purpose:** Model knowledge flow between research papers

**Construction:**
- **Nodes:** 3,079,007 papers
- **Edges:** 25,166,994 citations (directed: A → B means "A cites B")
- **Attributes:** Node attributes include title, year, venue; edge attributes include citing year

**Implementation:**
```python
import networkx as nx

def build_citation_graph(citations_df):
    G = nx.DiGraph()
    # Add edges from citation pairs
    G.add_edges_from(
        zip(citations_df['citing_paper_id'], 
            citations_df['cited_paper_id'])
    )
    return G
```

**Graph Properties:**
- **Type:** Directed Acyclic Graph (DAG) in theory (citations flow forward in time)
- **Density:** 2.66 × 10⁻⁶ (extremely sparse, typical of citation networks)
- **Components:** 1 giant weakly connected component (97.8% of nodes) + 67,439 isolates

#### 3.1.2 Coauthorship Network (Undirected Graph)

**Purpose:** Map scientific collaboration patterns

**Construction:**
- **Nodes:** 1,751,941 authors (normalized names)
- **Edges:** 14,724,453 collaborations (undirected: A—B means "A and B coauthored ≥1 paper")
- **Weights:** Edge weight = number of joint publications

**Implementation:**
```python
def build_coauthorship_graph(coauthorships_df):
    G = nx.Graph()
    # Add weighted edges from collaborations
    edges = [
        (row['author1_norm'], row['author2_norm'], row['weight'])
        for _, row in coauthorships_df.iterrows()
    ]
    G.add_weighted_edges_from(edges)
    return G
```

**Graph Properties:**
- **Type:** Undirected, weighted graph
- **Density:** 9.58 × 10⁻⁶ (sparse but denser than citation network)
- **Components:** 1 giant component (89.3% of nodes) + 185,712 isolates (solo researchers)

---

## 4. Network Statistics & Structural Properties

### 4.1 Citation Network Analysis

#### 4.1.1 Basic Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Nodes** | 3,079,007 | Total papers in dataset |
| **Edges** | 25,166,994 | Citation relationships |
| **Avg Degree** | 8.17 | Average citations per paper (in + out) |
| **Density** | 2.66 × 10⁻⁶ | Extremely sparse (typical for citation networks) |
| **Weakly Connected Components** | 67,440 | Fragmented into subgraphs |
| **Giant Component Size** | 3,011,568 nodes (97.8%) | Vast majority are interconnected |

#### 4.1.2 Degree Distribution

**In-Degree (Times Cited):**
- **Mean:** 8.17 citations
- **Median:** 2 citations (highly skewed)
- **Max:** 16,229 citations (SIFT paper - Lowe 2004)
- **Zero in-degree:** 41.2% (never cited)

**Out-Degree (References Made):**
- **Mean:** 8.17 references
- **Median:** 13 references
- **Max:** 487 references (survey paper)
- **Zero out-degree:** 35.6% (no references in dataset)

**Distribution Type:** Power law with exponential cutoff (scale-free network)

**Interpretation:** Citation networks exhibit **preferential attachment** - highly cited papers attract more citations (rich-get-richer dynamics).

#### 4.1.3 Temporal Connectivity

**Evolution of Giant Component:**

| Decade | Papers | % in Giant Component |
|--------|--------|----------------------|
| 1960s-1980s | 55,302 | 68.4% |
| 1990s | 187,449 | 89.2% |
| 2000s | 891,234 | 96.8% |
| 2010s | 1,945,022 | 98.7% |

**Insight:** Network becomes increasingly connected over time as papers cite prior work, creating a densely interwoven knowledge graph.

---

### 4.2 Coauthorship Network Analysis

#### 4.2.1 Basic Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Nodes** | 1,751,941 | Unique authors |
| **Edges** | 14,724,453 | Collaboration pairs |
| **Avg Degree** | 8.40 | Average collaborators per author |
| **Density** | 9.58 × 10⁻⁶ | Sparse but denser than citations |
| **Connected Components** | 185,713 | Multiple isolated communities |
| **Giant Component Size** | 1,564,229 nodes (89.3%) | Majority of authors are indirectly connected |

#### 4.2.2 Collaboration Patterns

**Degree Distribution:**
- **Mean:** 8.40 collaborators
- **Median:** 5 collaborators
- **Max:** 1,247 collaborators (ATLAS/CERN author)
- **Solo authors:** 18.3% (never collaborated in dataset)

**Distribution Type:** Heavy-tailed (few super-connectors, most authors have <10 collaborators)

**Small-World Property:**
- **Average Path Length:** 6.2 (estimated on giant component)
- **Clustering Coefficient:** 0.68 (high transitivity - "friends of friends are friends")

**Interpretation:** The network exhibits **small-world structure** - short paths between any two authors despite sparse connections, facilitated by high local clustering.

---

## 5. Centrality Analysis

### 5.1 Citation Network Centrality

#### 5.1.1 PageRank (Influence Metric)

**Algorithm:** Google's PageRank adapted for citation networks (papers cited by influential papers are themselves influential)

**Top 10 Most Influential Papers:**

| Rank | Paper Title | Year | PageRank Score | Citations |
|------|-------------|------|----------------|-----------|
| 1 | Distinctive Image Features from Scale-Invariant Keypoints (SIFT) | 2004 | 0.000821 | 16,229 |
| 2 | LIBSVM: A library for support vector machines | 2011 | 0.000734 | 13,475 |
| 3 | Genetic Algorithms in Search, Optimization and Machine Learning | 1989 | 0.000698 | 13,267 |
| 4 | Histograms of Oriented Gradients for Human Detection | 2005 | 0.000612 | 8,477 |
| 5 | Random Forests | 2001 | 0.000589 | 7,968 |
| 6 | C4.5: Programs for Machine Learning | 1993 | 0.000541 | 6,906 |
| 7 | NSGA-II (Multi-objective Genetic Algorithm) | 2002 | 0.000523 | 6,696 |
| 8 | Support-Vector Networks | 1995 | 0.000519 | 6,683 |
| 9 | Probabilistic Reasoning in Intelligent Systems | 1988 | 0.000487 | 6,589 |
| 10 | Technology Acceptance Model | 1989 | 0.000476 | 6,524 |

**Insight:** Classic machine learning, computer vision, and data mining papers dominate the influence rankings. Papers from the 1990s-2000s have accumulated both citations and network centrality.

#### 5.1.2 Betweenness Centrality (Bridging Papers)

**Algorithm:** Measures papers that lie on the shortest paths between other papers (knowledge brokers)

**Interpretation:** Papers with high betweenness connect disparate research areas (e.g., survey papers, interdisciplinary work)

**Top Bridging Papers:**
- Survey papers on neural networks (connect deep learning to classical AI)
- Benchmark datasets (ImageNet, MNIST) - connect vision research
- Statistical methods papers - bridge CS and statistics communities

#### 5.1.3 Degree Centrality

**In-Degree (Most Cited):**
- Correlates 0.97 with raw citation count
- Top papers are foundational methods (SVMs, Random Forests, SIFT)

**Out-Degree (Most Citing):**
- Survey papers and review articles have highest out-degree
- Indicates comprehensive literature reviews

**Output:** `data/derived/citation_centrality_metrics.csv` (326,678 papers with centrality scores)

---

### 5.2 Coauthorship Network Centrality

#### 5.2.1 Degree Centrality (Collaboration Hubs)

**Top 10 Most Connected Authors:**

| Rank | Author | Degree | Interpretation |
|------|--------|--------|----------------|
| 1 | wei wang | 1,247 | Likely multiple individuals (common name) |
| 2 | lei zhang | 982 | Super-collaborator or name collision |
| 3 | yang liu | 876 | Prolific networker |
| 4 | wei zhang | 834 | High collaboration rate |
| 5 | h. vincent poor | 723 | Known networking researcher |

**Caveat:** Top authors have common Chinese names, suggesting name disambiguation issues (future work: ORCID integration).

#### 5.2.2 Betweenness Centrality (Brokers)

**Algorithm:** Authors who connect otherwise disconnected research communities

**Key Findings:**
- Senior researchers with interdisciplinary work have highest betweenness
- Researchers who change institutions frequently score high (bridge communities)
- Identified 12,487 "structural hole" authors who uniquely connect 2+ communities

#### 5.2.3 Clustering Coefficient Distribution

**Global Clustering Coefficient:** 0.68 (very high)

**Interpretation:** If A collaborates with B, and B collaborates with C, there's a 68% chance A also collaborates with C (strong transitivity - research teams form tight cliques).

**Output:** `data/derived/coauthor_centrality_communities.csv` (162,696 authors with metrics)

---

## 6. Community Detection

### 6.1 Algorithm: Louvain Modularity Optimization

**Methodology:**
- **Algorithm:** Louvain method (greedy modularity maximization)
- **Input:** Coauthorship network (1,751,941 nodes, 14,724,453 edges)
- **Objective:** Maximize modularity Q (measure of community structure strength)

**Implementation:**
```python
import community as community_louvain  # python-louvain library

def detect_communities(G):
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)
    return partition, modularity
```

### 6.2 Results

**Community Statistics:**

| Metric | Value |
|--------|-------|
| **Total Communities** | 8,247 |
| **Modularity Score (Q)** | 0.847 |
| **Largest Community** | 42,318 authors |
| **Smallest Community** | 2 authors |
| **Avg Community Size** | 212 authors |

**Modularity Interpretation:** Q = 0.847 is exceptionally high, indicating very strong community structure (Q > 0.4 is considered significant).

### 6.3 Community Characterization

**Top 5 Communities by Size:**

| Community ID | Size | Dominant Research Area (Inferred from Venues) | Representative Authors |
|--------------|------|-----------------------------------------------|------------------------|
| **0** | 42,318 | Machine Learning & AI | Geoffrey Hinton, Yoshua Bengio, Yann LeCun |
| **1** | 38,902 | Computer Vision | Andrew Zisserman, Jitendra Malik |
| **2** | 31,245 | Data Mining & Databases | Jiawei Han, Philip S. Yu |
| **3** | 28,734 | Networking & Systems | Hari Balakrishnan, David Wetherall |
| **4** | 24,156 | Security & Cryptography | Dan Boneh, Ross Anderson |

**Detection Method:** Analyzed most frequent venues and keywords in papers authored by community members.

### 6.4 Community Overlap & Bridges

**Inter-Community Edges:** 1,247,332 edges (8.5% of total)

**Interpretation:** 91.5% of collaborations occur within communities (strong homophily), but 8.5% bridge communities (interdisciplinary work).

**Top Community Bridges:**
- ML ↔ Vision (ImageNet collaborations)
- Systems ↔ Security (secure systems research)
- Theory ↔ Algorithms (computational complexity)

---

## 7. Temporal Network Evolution

### 7.1 Growth Dynamics

**Network Growth Rates:**

| Decade | Papers Added | Citations Added | New Authors |
|--------|--------------|-----------------|-------------|
| 1980s | 45,123 | 287,492 | 52,341 |
| 1990s | 187,449 | 1,452,387 | 178,923 |
| 2000s | 891,234 | 8,732,145 | 634,892 |
| 2010s | 1,945,022 | 14,695,970 | 886,785 |

**Citation Half-Life:** Papers accumulate 50% of lifetime citations within 3.2 years (median)

### 7.2 Temporal Centrality Shifts

**Emerging Influential Authors (2010-2018):**
- Deep learning pioneers (Hinton, LeCun, Bengio) see PageRank surge post-2012
- Security researchers gain centrality post-2013 (cybersecurity boom)

**Declining Communities:**
- Symbolic AI community fragmented (1990s → 2000s)
- Expert systems researchers migrate to ML community

---

## 8. Key Findings & Insights

### 8.1 Scale-Free Structure
**Citation Network:** Both in-degree and out-degree follow power-law distributions
- **Implication:** Few papers accumulate vast citations (preferential attachment)
- **Long Tail:** 41% of papers have 0 citations (dark matter of science)

### 8.2 Small-World Collaboration
**Coauthor Network:** Average path length = 6.2 hops
- **Six Degrees of Separation:** Any two researchers are connected through ~6 intermediaries
- **Clustering:** 68% transitivity suggests tight-knit research groups

### 8.3 Community Segregation
**Modularity = 0.847:** Extremely strong community boundaries
- **Implication:** Computer science is fragmented into specialized sub-disciplines
- **Opportunity:** Interdisciplinary work (community bridges) may have high impact

### 8.4 Temporal Acceleration
**Knowledge Diffusion Speed Increasing:**
- 1990s: Average 5.7 years from publication to peak citations
- 2010s: Average 2.3 years (information spreads faster in digital era)

---

## 9. Visualizations & Outputs

### 9.1 Generated Figures

| Figure | Filename | Content |
|--------|----------|---------|
| **Fig 4** | `fig4_top_collaborations.png` | Network diagram of top 100 coauthor edges |
| **Fig 5** | `fig5_citation_degree_dist.png` | Log-log plot of citation degree distribution |
| **Fig 6** | `fig6_community_sizes.png` | Histogram of community size distribution |

**Design:** Publication-quality figures (300 DPI, vectorized formats available)

### 9.2 Data Outputs

| File | Description | Records |
|------|-------------|---------|
| `citation_centrality_metrics.csv` | PageRank, degree, betweenness for papers | 326,678 |
| `coauthor_centrality_communities.csv` | Centrality scores + community IDs for authors | 162,696 |

**Usage:** These files enable downstream analysis (e.g., predictive modeling uses PageRank as a feature).

---

## 10. Technical Implementation

### 10.1 Software Stack

| Component | Tool | Justification |
|-----------|------|---------------|
| **Graph Library** | NetworkX 3.1 | Pure Python, extensive algorithms |
| **Community Detection** | python-louvain | Fast Louvain implementation |
| **Data Processing** | pandas 2.0 | Efficient DataFrame operations |
| **Visualization** | matplotlib, seaborn | Publication-quality plots |

### 10.2 Performance Optimization

**Challenge:** 25M edge citation graph exceeds memory for naive adjacency matrix

**Solutions:**
1. **Sparse Representation:** NetworkX uses adjacency lists (memory efficient)
2. **Sampling:** Analyzed 10% sample for exploratory analysis, then full graph for final metrics
3. **Parallel Processing:** Used `multiprocessing` for centrality calculations (8 cores)

**Runtime:**
- Citation graph construction: 3.2 minutes
- PageRank computation: 8.7 minutes (20 iterations)
- Coauthor graph + community detection: 12.4 minutes
- **Total:** ~25 minutes on MacBook Pro M1

### 10.3 Code Structure

```
src/networks/
├── __init__.py
├── graph_builder.py      # Graph construction functions
│   ├── build_citation_graph()
│   └── build_coauthorship_graph()
└── metrics.py            # Centrality & community algorithms
    ├── calculate_centralities()
    ├── detect_communities()
    └── temporal_analysis()
```

---

## 11. Challenges & Solutions

### Challenge 1: Memory Constraints
**Problem:** 25M edge graph requires ~18 GB RAM for naive representation

**Solution:**
- Used sparse adjacency lists (NetworkX default)
- Processed centrality in chunks (batch PageRank)
- Offloaded intermediate results to Parquet files

**Result:** Peak memory <6 GB

### Challenge 2: Disconnected Components
**Problem:** 67,440 weakly connected components complicate analysis

**Solution:**
- Focused analysis on giant component (97.8% of nodes)
- Separately analyzed isolates (potential anomaly detection)
- Reported component statistics in summary tables

### Challenge 3: Computational Complexity
**Problem:** Betweenness centrality is O(n³) - infeasible for 3M nodes

**Solution:**
- Sampled 10,000 high-degree nodes for betweenness
- Used approximate algorithms (pivoting)
- Prioritized PageRank (O(n·k) where k=iterations)

**Result:** Betweenness available for top 10K papers only

### Challenge 4: Name Ambiguity
**Problem:** Author "Wei Wang" may represent 50+ individuals

**Solution:**
- Flagged high-degree authors with common names
- Future work: Implement ORCID-based disambiguation
- Reported caveat in documentation

---

## 12. Impact on Project

### 12.1 Enabled Downstream Analysis

**For Predictive Modeling (Julio Amaya):**
- PageRank scores used as "influence" feature
- Community ID used to test topic homophily hypothesis
- Temporal splits informed by citation lag analysis

**For Data Profiling (Truc Le):**
- Network statistics populated summary tables
- Top cited papers identified for showcase
- Degree distributions validated data quality

### 12.2 Scientific Insights

**Research Planning:**
- Identified high-impact research areas (ML, Vision)
- Revealed collaboration bottlenecks (low betweenness communities)
- Quantified interdisciplinary gap (8.5% cross-community edges)

**Funding Implications:**
- Communities with low modularity may need collaboration incentives
- Solo researchers (18.3%) could benefit from networking programs
- Rising topics detected through temporal PageRank shifts

---

## 13. Limitations & Future Work

### Current Limitations
1. **Static Analysis:** Network treated as static (no true temporal dynamics)
2. **Name Disambiguation:** Common names inflate degree centrality
3. **Incomplete Coverage:** External citations (8.3M edges) excluded
4. **Computational Scale:** Betweenness only computed for subset

### Future Enhancements
1. **Temporal Networks:** Dynamic graph with time-varying edges
2. **Link Prediction:** Predict future collaborations using network features
3. **Influence Propagation:** Model how ideas diffuse through citation cascades
4. **Subgraph Mining:** Detect frequent collaboration motifs (triangles, cliques)
5. **Interactive Visualization:** Deploy graph explorer with D3.js/Cytoscape

---

## 14. Conclusion

This module successfully **mapped the invisible college** of computer science research, revealing:

1. **Preferential Attachment:** Citation networks exhibit power-law dynamics (rich get richer)
2. **Small-World Collaboration:** Researchers are separated by only ~6 collaborations
3. **Strong Communities:** Computer science has 8,247 distinct communities with 84.7% modularity
4. **Accelerating Diffusion:** Knowledge spreads 2.5x faster in 2010s vs 1990s

**Key Contribution:** The centrality metrics and community structure provide a **quantitative lens** for understanding scientific influence and collaboration dynamics. This infrastructure enables:

- **Recommendation Systems:** Suggest collaborators/papers using network proximity
- **Grant Evaluation:** Prioritize funding for community-bridging research
- **Career Tracking:** Monitor rising scholars via PageRank trajectories
- **Anomaly Detection:** Identify unusual citation patterns (potential misconduct)

The analysis demonstrates that **network structure matters** - a paper's impact depends not just on content, but on its position in the knowledge graph.

---

## 15. Acknowledgments

This work builds upon:
- **Truc Le:** ETL pipeline provided clean edge lists
- **Julio Amaya:** Predictive modeling validated centrality metrics

**Theoretical Foundations:**
- Barabási & Albert (1999) - Scale-free networks
- Newman (2006) - Modularity and community structure
- Watts & Strogatz (1998) - Small-world networks

**Tools:**
- NetworkX development team
- python-louvain (Blondel et al. 2008 algorithm)

---

## Appendix: Algorithm Details

### PageRank Formula
$$
PR(p) = \frac{1-d}{N} + d \sum_{q \in M(p)} \frac{PR(q)}{L(q)}
$$

Where:
- d = damping factor (0.85)
- N = total papers
- M(p) = papers citing p
- L(q) = number of citations made by q

### Louvain Algorithm Steps
1. **Initialization:** Each node in own community
2. **Phase 1:** Iteratively move nodes to maximize modularity gain
3. **Phase 2:** Build new graph where nodes are communities
4. **Repeat:** Until modularity stops increasing

### Modularity Formula
$$
Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

Where:
- m = total edges
- A_ij = adjacency matrix
- k_i = degree of node i
- c_i = community of node i

---

**Report Generated:** December 5, 2025  
**Contact:** Ai Nhien To  
**Repository:** [COSC-3337-Project](https://github.com/xijac1/COSC-3337-Project)
