import networkx as nx
import pandas as pd

def build_citation_graph(citations_df):
    """
    Builds a directed citation graph from the citations DataFrame.
    Nodes: Papers
    Edges: Citation (Paper A -> Paper B)
    """
    G = nx.DiGraph()
    # TODO: Add edges from dataframe
    # G.add_edges_from(...)
    return G

def build_coauthorship_graph(authorships_df):
    """
    Builds an undirected, weighted co-authorship graph.
    Nodes: Authors
    Edges: Co-authored a paper together (Weight = number of papers)
    """
    G = nx.Graph()
    # TODO: Logic to create pairwise combinations of authors per paper
    return G
