"""
Centrality and network metrics calculation utilities.
"""

import networkx as nx
import pandas as pd


def calculate_centralities(graph, graph_type='directed'):
    """
    Calculate centrality metrics for a graph.
    
    Parameters:
    -----------
    graph : nx.Graph or nx.DiGraph
        Input network graph
    graph_type : str
        'directed' for DiGraph (calculates in-degree, out-degree, PageRank, betweenness)
        'undirected' for Graph (calculates degree, betweenness)
    
    Returns:
    --------
    metrics_df : pd.DataFrame
        DataFrame with node IDs and centrality metrics
    """
    
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()
    
    nodes = list(graph.nodes())
    
    if graph_type == 'directed':
        # Citation graph metrics
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())
        pagerank = nx.pagerank(graph)
        betweenness = nx.betweenness_centrality(graph, k=min(1000, graph.number_of_nodes()))
        
        metrics_df = pd.DataFrame({
            'node': nodes,
            'in_degree': [in_degree.get(n, 0) for n in nodes],
            'out_degree': [out_degree.get(n, 0) for n in nodes],
            'pagerank': [pagerank.get(n, 0) for n in nodes],
            'betweenness': [betweenness.get(n, 0) for n in nodes]
        })
    
    else:  # undirected
        # Co-authorship graph metrics
        degree = dict(graph.degree())
        betweenness = nx.betweenness_centrality(graph, k=min(1000, graph.number_of_nodes()))
        
        metrics_df = pd.DataFrame({
            'node': nodes,
            'degree': [degree.get(n, 0) for n in nodes],
            'betweenness': [betweenness.get(n, 0) for n in nodes]
        })
    
    return metrics_df


def get_graph_stats(graph, graph_type='directed'):
    """
    Get basic statistics about a graph.
    
    Parameters:
    -----------
    graph : nx.Graph or nx.DiGraph
    graph_type : str
    
    Returns:
    --------
    stats : dict
        Dictionary with graph statistics
    """
    
    stats = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
    }
    
    if graph_type == 'directed':
        # Additional stats for directed graphs
        if graph.number_of_nodes() > 0:
            num_weakly_connected = nx.number_weakly_connected_components(graph)
            largest_wcc = max(nx.weakly_connected_components(graph), key=len)
            stats['num_weakly_connected_components'] = num_weakly_connected
            stats['largest_wcc_size'] = len(largest_wcc)
    else:
        # Additional stats for undirected graphs
        if graph.number_of_nodes() > 0:
            num_connected = nx.number_connected_components(graph)
            largest_cc = max(nx.connected_components(graph), key=len)
            stats['num_connected_components'] = num_connected
            stats['largest_cc_size'] = len(largest_cc)
    
    return stats


def detect_communities(G):
    """
    Detects communities using Louvain or other algorithms.
    """
    # TODO: Implement community detection
    return {}
