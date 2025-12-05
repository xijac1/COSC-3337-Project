"""
Graph construction utilities for citation and co-authorship networks.
"""

import networkx as nx
import pandas as pd


def build_citation_graph(citations_df):
    """
    Build a directed citation graph from citations DataFrame.
    
    Nodes: Papers (paper IDs)
    Edges: Directed edge from src_id to dst_id indicates citation
    
    Parameters:
    -----------
    citations_df : pd.DataFrame
        Must contain columns: 'src_id' (citing paper), 'dst_id' (cited paper)
    
    Returns:
    --------
    G : nx.DiGraph
        Directed graph where edge (src → dst) means src_id cites dst_id
    """
    if citations_df.empty:
        return nx.DiGraph()
    
    G = nx.from_pandas_edgelist(
        citations_df,
        source='src_id',
        target='dst_id',
        create_using=nx.DiGraph()
    )
    
    return G


def build_coauthorship_graph(coauthorships_df):
    """
    Build an undirected co-authorship graph from coauthorships DataFrame.
    
    Nodes: Authors (normalized author names)
    Edges: Undirected edge between authors who collaborated
    
    Parameters:
    -----------
    coauthorships_df : pd.DataFrame
        Must contain columns: 'author1_norm', 'author2_norm' (normalized author names)
    
    Returns:
    --------
    G : nx.Graph
        Undirected graph where nodes are authors and edges connect co-authors
    """
    if coauthorships_df.empty:
        return nx.Graph()
    
    G = nx.from_pandas_edgelist(
        coauthorships_df,
        source='author1_norm',
        target='author2_norm',
        create_using=nx.Graph()
    )
    
    return G
